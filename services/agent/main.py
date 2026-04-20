"""Agent worker entrypoint.

Runs an HTTP server on :8001 that accepts dispatch requests from the API.
Each request spawns an isolated Pipecat pipeline for the given room.
When the caller leaves, the pipeline shuts down and the room is freed.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

from aiohttp import web
from livekit import api as lkapi
from loguru import logger

from pipecat.frames.frames import InterruptionTaskFrame, TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner

from env import MissingEnvError, require_env
from models import get_gigaam, get_whisper, load_gigaam, load_whisper
from overrides import PerSessionOverrides, resolve_greeting
from pipeline import AgentConfig, build_task

AGENT_IDENTITY = "agent-bot"
AGENT_TOKEN_TTL = datetime.timedelta(minutes=30)

# Shared config loaded once at startup.
_livekit_url: str = ""
_api_key: str = ""
_api_secret: str = ""
_base_config: dict[str, object] = {}

# Track active rooms to prevent double-dispatch.
_active_rooms: set[str] = set()


def _mint_agent_token(room: str) -> str:
    """Mint a LiveKit JWT for the agent with full publish/subscribe in the room."""
    return (
        lkapi.AccessToken(_api_key, _api_secret)
        .with_identity(f"{AGENT_IDENTITY}-{room}")
        .with_name("Assistant")
        .with_grants(
            lkapi.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(AGENT_TOKEN_TTL)
        .to_jwt()
    )


async def _run_pipeline(room: str, overrides: PerSessionOverrides) -> None:
    """Run a Pipecat pipeline for one room until the caller leaves."""
    logger.info(
        "Spawning pipeline room={room} npc={npc} lang={lang} voice={voice} model={model}",
        room=room,
        npc=overrides.npc_id or "-",
        lang=overrides.effective_language,
        voice=overrides.voice or "default",
        model=overrides.llm_model or "default",
    )
    # The caller (handle_dispatch) claims the slot in _active_rooms BEFORE
    # scheduling this task — otherwise two concurrent dispatches for the
    # same room could both pass the guard check and spawn duplicate
    # pipelines. We only own the discard() in finally.
    try:
        cfg = AgentConfig(
            livekit_url=_livekit_url,
            livekit_token=_mint_agent_token(room),
            room_name=room,
            **_base_config,
        )

        task, transport = build_task(
            cfg,
            whisper_model=get_whisper(),
            gigaam_model=get_gigaam(),
            overrides=overrides,
        )
        greeting = resolve_greeting(overrides.persona, overrides.effective_language)

        @transport.event_handler("on_first_participant_joined")
        async def _on_join(_transport: object, participant_id: str) -> None:
            logger.info("Participant joined room={room}: {pid}", room=room, pid=participant_id)
            await transport.send_message(
                json.dumps({"role": "assistant", "text": greeting}, ensure_ascii=False)
            )
            await task.queue_frame(TTSSpeakFrame(greeting))

        @transport.event_handler("on_participant_left")
        async def _on_leave(_transport: object, participant_id: str, _reason: object) -> None:
            logger.info("Participant left room={room}: {pid}", room=room, pid=participant_id)
            await task.queue_frame(InterruptionTaskFrame())

        await PipelineRunner().run(task)
        logger.info("Pipeline exited for room={room}", room=room)
    except Exception:
        logger.exception("Pipeline failed for room={room}", room=room)
    finally:
        _active_rooms.discard(room)


async def handle_dispatch(request: web.Request) -> web.Response:
    """POST /dispatch — spawn a pipeline for the requested room."""
    body = await request.json()
    room = body.get("room")
    if not room:
        return web.json_response({"error": "room is required"}, status=400)
    if room in _active_rooms:
        return web.json_response({"error": "room already active"}, status=409)

    # Claim the slot synchronously — before any await or create_task yields
    # to the event loop. Otherwise a burst of identical dispatches could
    # all pass the guard above before any pipeline adds itself to the set.
    _active_rooms.add(room)
    overrides = PerSessionOverrides.from_dispatch(body)
    asyncio.create_task(_run_pipeline(room, overrides))
    return web.json_response({"status": "dispatched", "room": room})


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"status": "ok", "active_rooms": len(_active_rooms)})


def main() -> None:
    global _livekit_url, _api_key, _api_secret, _base_config

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    try:
        _livekit_url = require_env("LIVEKIT_URL")
        _api_key = require_env("LIVEKIT_API_KEY")
        _api_secret = require_env("LIVEKIT_API_SECRET")

        stt_stack = os.environ.get("STT_STACK", "gigaam").lower()
        if stt_stack not in {"gigaam", "whisper"}:
            raise MissingEnvError(
                f"STT_STACK={stt_stack!r} invalid; expected 'gigaam' or 'whisper'"
            )

        _base_config = {
            "vllm_base_url": require_env("VLLM_BASE_URL"),
            "vllm_model": require_env("VLLM_MODEL", "qwen-7b"),
            "tts_voice": require_env("TTS_VOICE", "ru_RU-denis-medium"),
            "vllm_api_key": require_env("VLLM_API_KEY", "not-used"),
            "piper_voices_dir": Path(require_env("PIPER_VOICES_DIR", "/home/appuser/.cache/piper")),
            "stt_stack": stt_stack,
            # Optional transcript persistence. Both must be set (or both
            # left empty — require_env with "" default accepts unset as
            # empty rather than raising).
            "api_base_url": os.environ.get("API_BASE_URL", ""),
            "agent_internal_token": os.environ.get("AGENT_INTERNAL_TOKEN", ""),
        }
    except MissingEnvError as exc:
        logger.error(str(exc))
        sys.exit(2)

    # Pre-load both STT models at startup. GigaAM is the default (see
    # docs/experiments/2026-04-19-stt-ab/phase-a-results.md); Whisper stays
    # loaded so flipping STT_STACK=whisper on restart is zero-latency
    # rather than paying the 25s Whisper load at first dispatch. Hard-fail
    # on either load — a half-loaded agent is worse than a down agent
    # because the STT_STACK env toggle would silently not work.
    load_whisper()
    try:
        load_gigaam()
    except Exception:  # noqa: BLE001 — intentional hard-fail at startup
        logger.exception("GigaAM pre-load failed — refusing to start agent")
        sys.exit(3)
    logger.info("STT stack: {stack}", stack=stt_stack)

    app = web.Application()
    app.router.add_post("/dispatch", handle_dispatch)
    app.router.add_get("/health", handle_health)

    logger.info("Agent dispatcher starting on :8001")
    web.run_app(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

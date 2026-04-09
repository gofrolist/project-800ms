"""Agent worker entrypoint.

MVP model: one long-running agent joins a fixed demo room and stays. The API
mints caller tokens for the same room. Single-tenant, single-agent — good
enough to prove the latency target. Per-call dispatch is a follow-up.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
from pathlib import Path

from livekit import api as lkapi
from loguru import logger
import json

from pipecat.frames.frames import InterruptionTaskFrame, TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner

from env import MissingEnvError, require_env
from pipeline import AgentConfig, build_task

GREETING = "Привет! Чем могу помочь?"

AGENT_IDENTITY = "agent-bot"
AGENT_TOKEN_TTL = datetime.timedelta(hours=2)


def _mint_agent_token(api_key: str, api_secret: str, room: str) -> str:
    """Mint a LiveKit JWT for the agent with full publish/subscribe in the room."""
    return (
        lkapi.AccessToken(api_key, api_secret)
        .with_identity(AGENT_IDENTITY)
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


async def run() -> None:
    livekit_url = require_env("LIVEKIT_URL")
    api_key = require_env("LIVEKIT_API_KEY")
    api_secret = require_env("LIVEKIT_API_SECRET")
    room = require_env("LIVEKIT_ROOM", "demo")

    cfg = AgentConfig(
        livekit_url=livekit_url,
        livekit_token=_mint_agent_token(api_key, api_secret, room),
        room_name=room,
        vllm_base_url=require_env("VLLM_BASE_URL"),
        vllm_model=require_env("VLLM_MODEL", "mistral-7b"),
        tts_voice=require_env("TTS_VOICE", "ru_RU-denis-medium"),
        vllm_api_key=require_env("VLLM_API_KEY", "not-used"),
        piper_voices_dir=Path(require_env("PIPER_VOICES_DIR", "/home/appuser/.cache/piper")),
    )
    logger.info(
        "Agent joining room={room} model={model} voice={voice}",
        room=cfg.room_name,
        model=cfg.vllm_model,
        voice=cfg.tts_voice,
    )

    task, transport = build_task(cfg)

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(_transport, participant_id):  # noqa: ANN001
        logger.info(f"Participant joined: {participant_id}")
        await transport.send_message(
            json.dumps({"role": "assistant", "text": GREETING}, ensure_ascii=False)
        )
        await task.queue_frame(TTSSpeakFrame(GREETING))

    @transport.event_handler("on_participant_left")
    async def _on_leave(_transport, participant_id, _reason):  # noqa: ANN001
        logger.info(f"Participant left: {participant_id}")
        await task.queue_frame(InterruptionTaskFrame())

    await PipelineRunner().run(task)
    logger.info("Agent runner exited")


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except MissingEnvError as exc:
        logger.error(str(exc))
        sys.exit(2)


if __name__ == "__main__":
    main()

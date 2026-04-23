"""Agent worker entrypoint.

Runs an HTTP server on :8001 that accepts dispatch requests from the API.
Each request spawns an isolated Pipecat pipeline for the given room.
When the caller leaves, the pipeline shuts down and the room is freed.
"""

from __future__ import annotations

import asyncio
import datetime
import errno
import json
import os
import shutil
import sys
from pathlib import Path

from aiohttp import web
from livekit import api as lkapi
from loguru import logger

from pipecat.frames.frames import InterruptionTaskFrame, TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner

from env import MissingEnvError, require_env
from models import get_gigaam, load_gigaam, load_silero, load_xtts
from overrides import PerSessionOverrides, resolve_greeting
from pipeline import AgentConfig, build_task

AGENT_IDENTITY = "agent-bot"
AGENT_TOKEN_TTL = datetime.timedelta(minutes=30)

# Minimum free space at TTS_HOME before we even try the XTTS download.
# The weights themselves are ~1.8 GB; 2 GB leaves breathing room for the
# partial-download file and the final rename without hitting ENOSPC
# mid-way. If the cache volume has less than this, skip preload and
# degrade XTTS to "unavailable on this agent" rather than crashlooping
# the whole container (which would take Piper/Silero/Qwen3 down too).
_XTTS_MIN_FREE_BYTES = 2_000_000_000

# Shared config loaded once at startup.
_livekit_url: str = ""
_api_key: str = ""
_api_secret: str = ""
_base_config: dict[str, object] = {}

# Track active rooms to prevent double-dispatch.
_active_rooms: set[str] = set()

# Engines this process is ready to dispatch. Populated at startup from
# TTS_PRELOAD_ENGINES (see main()). Used by handle_dispatch to reject
# sessions targeting an engine the process isn't configured for — without
# this guard, such a dispatch returns 200 + LiveKit token, and the
# pipeline failure only surfaces as "user hears silence" much later.
_preload_engines: frozenset[str] = frozenset()


def _xtts_has_disk_space() -> bool:
    """Return True if TTS_HOME has enough free space for the XTTS download.

    Checks the filesystem free-bytes at the TTS_HOME directory (or its
    nearest existing ancestor on first boot, before coqui-tts has
    created the cache directory). Returns True if free bytes >=
    ``_XTTS_MIN_FREE_BYTES``, False otherwise. Logs a warning with the
    measured free space when the check fails so operators can correlate
    with downstream 409s on XTTS sessions.

    Returns True on any OSError from the disk_usage probe — err on the
    side of attempting the download and letting the real load_xtts
    surface a specific error, rather than silently skipping preload
    on spurious probe failures.
    """
    tts_home_env = os.environ.get("TTS_HOME")
    probe = Path(tts_home_env) if tts_home_env else Path.home() / ".local" / "share" / "tts"
    # Walk up to an existing ancestor so disk_usage doesn't fail on a
    # not-yet-created cache path.
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        logger.warning(
            "Could not probe disk space at {path}; attempting XTTS preload anyway",
            path=probe,
        )
        return True
    if free < _XTTS_MIN_FREE_BYTES:
        logger.warning(
            "XTTS cache {path} has {free_mb:.0f} MB free — below {min_mb:.0f} MB "
            "minimum for the 1.8 GB weight download",
            path=probe,
            free_mb=free / 1_000_000,
            min_mb=_XTTS_MIN_FREE_BYTES / 1_000_000,
        )
        return False
    return True


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

    # Early-validate the engine against this agent's preloaded set. Without
    # this guard, a dispatch with ``tts_engine=xtts`` on a deploy where
    # ``xtts`` isn't in ``TTS_PRELOAD_ENGINES`` would return 200 here, reach
    # ``build_tts_service`` inside ``_run_pipeline``, hit ``get_xtts()``'s
    # ``RuntimeError("XTTS model not loaded")``, get swallowed by the bare
    # ``except Exception``, and leave the caller in a live LiveKit room with
    # no agent. Failing fast at 409 surfaces the misconfiguration in the
    # caller's HTTP response instead.
    overrides = PerSessionOverrides.from_dispatch(body)
    requested_engine = overrides.tts_engine or str(_base_config.get("tts_engine", ""))
    if requested_engine and _preload_engines and requested_engine not in _preload_engines:
        logger.warning(
            "Dispatch refused: tts_engine={engine} not in preload set {preload}",
            engine=requested_engine,
            preload=sorted(_preload_engines),
        )
        return web.json_response(
            {
                "error": "engine not available on this agent",
                "requested_engine": requested_engine,
                "available_engines": sorted(_preload_engines),
            },
            status=409,
        )

    # Claim the slot synchronously — before any await or create_task yields
    # to the event loop. Otherwise a burst of identical dispatches could
    # all pass the guard above before any pipeline adds itself to the set.
    _active_rooms.add(room)
    asyncio.create_task(_run_pipeline(room, overrides))
    return web.json_response({"status": "dispatched", "room": room})


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"status": "ok", "active_rooms": len(_active_rooms)})


def main() -> None:
    global _livekit_url, _api_key, _api_secret, _base_config, _preload_engines

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    try:
        _livekit_url = require_env("LIVEKIT_URL")
        _api_key = require_env("LIVEKIT_API_KEY")
        _api_secret = require_env("LIVEKIT_API_SECRET")

        _base_config = {
            "vllm_base_url": require_env("VLLM_BASE_URL"),
            "vllm_model": require_env("VLLM_MODEL", "qwen-7b"),
            "tts_voice": require_env("TTS_VOICE", "ru_RU-denis-medium"),
            "vllm_api_key": require_env("VLLM_API_KEY", "not-used"),
            "piper_voices_dir": Path(require_env("PIPER_VOICES_DIR", "/home/appuser/.cache/piper")),
            # TTS engine selector — "piper" (default), "silero" (Unit 3),
            # "qwen3" (Unit 4). The factory in tts_factory.py raises
            # ValueError on unknown values when a pipeline is built.
            "tts_engine": require_env("TTS_ENGINE", "piper"),
            # Qwen3-TTS sidecar wiring. Optional at agent startup — the
            # factory validates non-empty at dispatch time only when
            # TTS_ENGINE=qwen3, so Piper/Silero deploys don't need to set
            # them. Use os.environ.get() instead of require_env() which
            # rejects empty-string defaults.
            "qwen3_base_url": os.environ.get("QWEN3_TTS_BASE_URL", ""),
            "qwen3_api_key": os.environ.get("QWEN3_TTS_API_KEY", ""),
            # Qwen3-specific voice name (e.g. "clone:<profile>" to use a
            # voice-library profile baked into the wrapper image). Takes
            # precedence over TTS_VOICE when the Qwen3 engine dispatches.
            # See tts_factory.py's qwen3 branch for resolution order.
            "qwen3_tts_voice": os.environ.get("QWEN3_TTS_VOICE", ""),
            # XTTS-specific voice name. Same "clone:<profile>" shape as
            # the Qwen3 voice but resolved against the local voice_library
            # mount (see XTTS_VOICE_LIBRARY_DIR below). Takes precedence
            # over TTS_VOICE when the XTTS engine dispatches; empty →
            # fall back to the generic TTS_VOICE (which must itself be
            # a ``clone:<profile>`` value for XTTS to work).
            "xtts_tts_voice": os.environ.get("XTTS_TTS_VOICE", ""),
            # Shared voice-library root. Same directory the Qwen3 wrapper
            # reads from — XTTS and Qwen3 can pick up the same profile
            # without duplicating reference audio. Default matches the
            # agent container's bind-mount target in docker-compose.yml.
            "xtts_voice_library_dir": Path(
                os.environ.get("XTTS_VOICE_LIBRARY_DIR", "/opt/voice_library")
            ),
            # Optional transcript persistence. Both must be set (or both
            # left empty — require_env with "" default accepts unset as
            # empty rather than raising).
            "api_base_url": os.environ.get("API_BASE_URL", ""),
            "agent_internal_token": os.environ.get("AGENT_INTERNAL_TOKEN", ""),
        }
    except MissingEnvError as exc:
        logger.error(str(exc))
        sys.exit(2)

    # Pre-load GigaAM at startup so the first dispatched room doesn't pay
    # the model-load cost on its first utterance. Hard-fail if the load
    # raises — a half-loaded agent is worse than a down agent.
    try:
        load_gigaam()
    except Exception:  # noqa: BLE001 — intentional hard-fail at startup
        logger.exception("GigaAM pre-load failed — refusing to start agent")
        sys.exit(3)

    # Conditional Silero pre-load. The `TTS_PRELOAD_ENGINES` env var is a
    # comma-separated list of engines the agent should be ready to serve
    # per-session (see PerSessionOverrides.tts_engine). Silero is the only
    # engine with an agent-side preload cost — Piper has none (loaded
    # lazily inside PiperTTSService) and Qwen3's model lives in the
    # sidecar container. Default: `TTS_ENGINE`'s value, preserving the
    # single-engine operator workflow. Set `TTS_PRELOAD_ENGINES=piper,silero,qwen3`
    # when the demo site's three-button selector needs all three ready
    # without a cold-load on first dispatch. Same hard-fail semantics as
    # GigaAM above.
    preload_engines = {
        e.strip()
        for e in os.environ.get(
            "TTS_PRELOAD_ENGINES",
            str(_base_config["tts_engine"]),
        ).split(",")
        if e.strip()
    }
    # Expose the preload set to handle_dispatch so sessions targeting an
    # engine this process isn't configured for fail fast with 409 instead
    # of producing a silent dead session after dispatch.
    _preload_engines = frozenset(preload_engines)
    if "silero" in preload_engines:
        try:
            load_silero()
        except Exception:  # noqa: BLE001 — intentional hard-fail at startup
            logger.exception("Silero pre-load failed — refusing to start agent")
            sys.exit(3)

    # XTTS pre-load. ~1.8 GB VRAM and ~15-30 s on L4; the only engine in
    # the set that ships both a weight download AND a torch.load step,
    # so "lazy on first dispatch" would make the first XTTS session wait
    # 30+ seconds for the greeting. Gated behind the same TTS_PRELOAD_ENGINES
    # env list as Silero — set ``TTS_PRELOAD_ENGINES=piper,silero,xtts``
    # (or include ``xtts`` in whatever comma list the demo deploy uses)
    # to have the agent ready to serve XTTS on first dispatch.
    if "xtts" in preload_engines:
        # Startup invariant check: XTTS is a zero-shot voice cloner and
        # every session needs a voice_library profile. Docker compose
        # will silently create an empty bind-mount target if the host
        # path is missing, so the first dispatch would raise inside
        # ``_resolve_voice_profile`` → get swallowed → silent dead session.
        # Hard-fail at boot with a clear message — a missing profile
        # mount is an operator misconfiguration, not a transient
        # resource issue, and won't self-heal on restart.
        voice_library_dir = _base_config.get("xtts_voice_library_dir")
        if isinstance(voice_library_dir, Path):
            profiles_dir = voice_library_dir / "profiles"
            if not profiles_dir.is_dir() or not any(profiles_dir.iterdir()):
                logger.error(
                    "XTTS preload requested but no profiles found at {path}. "
                    "Check the voice_library bind mount in docker-compose.yml.",
                    path=profiles_dir,
                )
                sys.exit(3)

        # Disk-space pre-check: XTTS downloads ~1.8 GB into the
        # hf_cache_agent volume. If free space is below the 2 GB
        # threshold, skip preload and drop xtts from the preload set.
        # The dispatch handler already returns 409 for engines missing
        # from _preload_engines, so sessions pick a different engine
        # instead of the whole agent crashlooping on ENOSPC.
        #
        # Rationale: disk-full is a resource constraint, not a code
        # bug — hard-failing takes Piper/Silero/Qwen3 down as
        # collateral. Graceful degrade keeps the other engines serving
        # while operators free space.
        if not _xtts_has_disk_space():
            logger.warning(
                "Skipping XTTS preload due to insufficient disk space. "
                "Sessions requesting tts_engine=xtts will receive 409 "
                "until the cache volume has free space.",
            )
            preload_engines.discard("xtts")
            _preload_engines = frozenset(preload_engines)
        else:
            try:
                load_xtts()
            except OSError as e:  # noqa: BLE001 — specific disk-full handling below
                if e.errno == errno.ENOSPC:
                    # Disk filled mid-download (the pre-check passed but
                    # something else consumed the free space between
                    # probe and download). Same degrade path as above.
                    logger.warning(
                        "XTTS preload failed with ENOSPC mid-download. "
                        "Dropping xtts from the preload set; sessions "
                        "requesting tts_engine=xtts will receive 409.",
                    )
                    preload_engines.discard("xtts")
                    _preload_engines = frozenset(preload_engines)
                else:
                    logger.exception("XTTS pre-load failed — refusing to start agent")
                    sys.exit(3)
            except Exception:  # noqa: BLE001 — intentional hard-fail at startup
                logger.exception("XTTS pre-load failed — refusing to start agent")
                sys.exit(3)

    app = web.Application()
    app.router.add_post("/dispatch", handle_dispatch)
    app.router.add_get("/health", handle_health)

    logger.info("Agent dispatcher starting on :8001")
    web.run_app(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

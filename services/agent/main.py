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

from livekit import api as lkapi
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner

from env import MissingEnvError, require_env
from pipeline import AgentConfig, build_task

AGENT_IDENTITY = "agent-bot"
# Short TTL — the agent restarts on Docker container restart and re-mints
# its own token via require_env() at startup. A leaked agent token can be
# replayed for the TTL window, so keep it tight. The LiveKit Go SDK that
# pipecat uses re-authenticates on reconnect using the original credentials.
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
        vllm_model=require_env("VLLM_MODEL", "qwen2.5-7b"),
        tts_voice=require_env("TTS_VOICE", "ru_RU-ruslan-medium"),
        vllm_api_key=require_env("VLLM_API_KEY", "not-used"),
    )
    logger.info(f"Agent joining room={cfg.room_name} model={cfg.vllm_model} voice={cfg.tts_voice}")

    task, transport = build_task(cfg)

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(_transport, participant_id):  # noqa: ANN001
        logger.info(f"Participant joined: {participant_id}")
        # Short bilingual-friendly greeting. Adjust to taste per user locale.
        await task.queue_frame(TTSSpeakFrame("Привет! Чем могу помочь?"))

    @transport.event_handler("on_participant_left")
    async def _on_leave(_transport, participant_id, _reason):  # noqa: ANN001
        logger.info(f"Participant left: {participant_id}")
        # Don't end the task — stay in the room waiting for the next caller.

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

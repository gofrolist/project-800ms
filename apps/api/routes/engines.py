"""GET /v1/engines — enumerate the TTS engines a given agent can dispatch.

Before this endpoint, callers had to probe-and-fail: ``POST /v1/sessions``
with ``tts_engine=xtts`` on a deploy where XTTS wasn't preloaded would
receive HTTP 409 (via the agent's dispatch guard) or, worse on older
agents, HTTP 201 followed by a silent dead session. Agent integrators
needed a way to introspect the deploy before committing to an engine.

Design: the API proxies the agent's ``GET :8001/engines`` endpoint and
layers static per-engine metadata (voice-format hints) on top. The
dynamic ``available`` flag comes from the agent's live ``_preload_engines``
set, so low-disk degrade or per-deploy TTS_PRELOAD_ENGINES config
flows through to the client.

The endpoint is authenticated via the same ``enforce_tenant_rate_limit``
dependency the session endpoints use — not because the engine list is
sensitive (it leaks only "which in-house engines are running"), but for
consistency with the rest of the /v1/* surface and to keep the rate
limiter from having an unauthenticated hole. Agents themselves don't
need to read their own ``/engines`` through this endpoint.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth import TenantIdentity
from rate_limit import enforce_tenant_rate_limit
from schemas import TtsEngine
from settings import settings

logger = logging.getLogger("project-800ms.api.engines")

router = APIRouter(prefix="/v1", tags=["engines"])


# Static per-engine metadata the agent itself doesn't know and shouldn't
# need to. Clients use these hints to decide how to format the ``voice``
# field on POST /v1/sessions.
#
# Order matches the frontend demo site's card layout (Piper, Silero,
# Qwen3, XTTS) — the API returns engines in this canonical order rather
# than sorting alphabetically, so UIs can render the list directly.
VoiceFormat = Literal[
    # Piper: voice-pack name, e.g. "ru_RU-denis-medium". Auto-downloaded
    # from HuggingFace on first use.
    "piper_voice_name",
    # Silero: speaker id within the v5_cis_base model. See
    # services/agent/silero_tts.py SileroSettings.speaker.
    "silero_speaker_id",
    # Qwen3: an OpenAI voice name ("alloy" / "echo" / ...) OR
    # "clone:<profile>" for voice-library profiles on Base variants.
    "openai_or_clone",
    # XTTS v2: always "clone:<profile>" — zero-shot voice cloning,
    # no preset catalog. See voice_library/README.md.
    "clone_only",
]


_ENGINE_METADATA: list[tuple[TtsEngine, VoiceFormat, str]] = [
    ("piper", "piper_voice_name", "Piper — CPU, baseline quality"),
    ("silero", "silero_speaker_id", "Silero v5 — GPU, Russian only"),
    ("qwen3", "openai_or_clone", "Qwen3-TTS sidecar — GPU, voice cloning"),
    ("xtts", "clone_only", "Coqui XTTS v2 — GPU, zero-shot voice cloning (CPML)"),
]


class EngineInfo(BaseModel):
    """Per-engine entry in the ``GET /v1/engines`` response."""

    id: TtsEngine = Field(description="Engine identifier accepted by POST /v1/sessions.tts_engine.")
    available: bool = Field(
        description=(
            "True if the agent has this engine preloaded and ready to serve. "
            "False when the agent is missing it (not in TTS_PRELOAD_ENGINES, "
            "or degraded out due to insufficient disk — see the agent's "
            "XTTS disk-space pre-check)."
        )
    )
    default: bool = Field(
        description=(
            "True if this engine is the agent's fallback when a session "
            "omits tts_engine. Matches the agent's TTS_ENGINE env."
        )
    )
    voice_format: VoiceFormat = Field(
        description=(
            "How to shape the voice field on POST /v1/sessions for this "
            "engine. See the VoiceFormat docstring for meaning of each value."
        )
    )
    label: str = Field(
        description="Human-readable description suitable for UI display.",
    )


class EnginesResponse(BaseModel):
    """Response body for ``GET /v1/engines``."""

    engines: list[EngineInfo] = Field(
        description=(
            "All engines the API knows about, each flagged with its "
            "agent-side availability on this deploy."
        )
    )


@router.get(
    "/engines",
    response_model=EnginesResponse,
    summary="List available TTS engines on this deploy",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Agent is down or unreachable — the API cannot report engine availability.",
        },
    },
)
async def list_engines(
    _identity: TenantIdentity = Depends(enforce_tenant_rate_limit),
) -> EnginesResponse:
    """Report which TTS engines are available on this agent deploy.

    Proxies the agent's ``GET :8001/engines`` and layers static
    ``voice_format`` + ``label`` metadata per engine. The dynamic
    ``available`` + ``default`` fields come from the agent.

    Raises 503 if the agent is unreachable or returns a malformed
    response — the client should retry later rather than assume the
    static engine list is authoritative.
    """
    # settings.agent_dispatch_url is the agent's base URL (same one
    # sessions.py hits for /dispatch). Strip any trailing slash so the
    # /engines path doesn't double up.
    agent_url = str(settings.agent_dispatch_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{agent_url}/engines")
    except httpx.HTTPError as exc:
        logger.warning("Agent /engines unreachable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is unreachable; cannot enumerate engines",
        ) from exc

    if resp.status_code != 200:
        logger.warning(
            "Agent /engines returned non-200: %s body=%s",
            resp.status_code,
            resp.text[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent /engines returned {resp.status_code}",
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent /engines returned invalid JSON",
        ) from exc

    # Pydantic has already validated the request dependency chain; here
    # we normalize the agent's response into the API-level contract.
    # Missing fields are treated as "nothing available" — fail closed.
    available_raw = payload.get("available", [])
    available_set = set(available_raw) if isinstance(available_raw, list) else set()
    default_engine = payload.get("default", "")

    engines = [
        EngineInfo(
            id=engine_id,
            available=engine_id in available_set,
            default=engine_id == default_engine,
            voice_format=fmt,
            label=label,
        )
        for engine_id, fmt, label in _ENGINE_METADATA
    ]
    return EnginesResponse(engines=engines)

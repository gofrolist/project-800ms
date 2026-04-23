"""Pydantic request/response schemas for /v1/* endpoints.

Keeping schemas separate from route handlers lets generated OpenAPI clients
import just the types, and makes it easy to spot breaking shape changes in
code review.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Whitelisted TTS engines the agent knows how to dispatch. Mirrors
# services/agent/overrides.py::_VALID_TTS_ENGINES and pipeline.py's
# _VALID_TTS_ENGINES — kept in sync by hand because the API and agent
# live in separate Python environments (3.14 vs 3.12).
TtsEngine = Literal["piper", "silero", "qwen3", "xtts"]


class CreateSessionRequest(BaseModel):
    """Game-client payload for opening a new voice session.

    Every field is optional because the simplest integration (a demo page)
    just wants a token and doesn't care about persona routing. Game clients
    fill in user_id/npc_id/persona/context for the features that need them.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(
        default=None,
        max_length=128,
        description="Stable caller identifier in the tenant's system.",
    )
    npc_id: str | None = Field(
        default=None,
        max_length=128,
        description="Which NPC / persona this session targets.",
    )
    persona: dict[str, Any] | None = Field(
        default=None,
        description="Persona JSON — system prompt, backstory, knobs.",
    )
    voice: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Provider-specific voice id. piper: e.g. 'ru_RU-denis-medium'. "
            "silero: e.g. 'v5_cis_base'. qwen3: one of the OpenAI voice names "
            "(alloy, echo, fable, nova, onyx, shimmer) or 'clone:<profile>'. "
            "xtts: 'clone:<profile>' (zero-shot voice cloning only). Defaults "
            "to the TTS_VOICE env var."
        ),
    )

    language: str | None = Field(
        default=None,
        max_length=16,
        description="BCP-47 locale hint for STT + LLM output.",
    )
    llm_model: str | None = Field(
        default=None,
        max_length=128,
        description="Override LLM model (falls back to tenant/agent default).",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary game-state context passed to the agent.",
    )
    tts_engine: TtsEngine | None = Field(
        default=None,
        description=(
            "TTS engine for this session — one of 'piper', 'silero', 'qwen3', "
            "'xtts'. Falls back to the agent's TTS_ENGINE env default when "
            "omitted. Used by the demo site's engine selector to route each "
            "session to a different backend without restarting the agent. "
            "Note: 'xtts' uses Coqui XTTS v2 weights released under the "
            "Coqui Public Model License (non-commercial); verify licensing "
            "posture before enabling on paid deployments."
        ),
    )

    def validate_engine_voice_combo(self) -> None:
        """Reject ``tts_engine=xtts`` paired with a non-``clone:`` voice.

        XTTS v2 is a pure voice-cloning engine — the agent's
        ``_resolve_voice_profile`` raises ``ValueError`` if the voice doesn't
        start with ``clone:``, but that error happens inside the asynchronous
        ``_run_pipeline`` task AFTER the dispatch handler has already
        returned 200 and the API has already returned 201 to the caller.
        Result: caller receives a valid LiveKit token for a room where the
        agent never arrives, and hears silence with no HTTP error.

        Called from the route handler at request time so a mismatch
        surfaces as 422 before any DB row is written or dispatch is
        attempted. (We picked imperative validation over a Pydantic
        ``field_validator`` because the project's middleware stack —
        ``BaseHTTPMiddleware`` + slowapi + Prometheus — has an edge case
        where a ValidationError raised from a custom validator was
        surfacing as 500 instead of 422.)
        """
        if (
            self.tts_engine == "xtts"
            and self.voice is not None
            and not self.voice.startswith("clone:")
        ):
            raise ValueError(
                "tts_engine='xtts' requires voice to start with 'clone:<profile>' — "
                "XTTS v2 is a voice-cloning engine and has no preset voice names. "
                f"Got voice={self.voice!r}."
            )


class CreateSessionResponse(BaseModel):
    """What the browser / game client needs to join the LiveKit room."""

    session_id: str
    room: str
    identity: str
    token: str
    url: str


class SessionDetails(BaseModel):
    """Read model for GET /v1/sessions/{room}."""

    session_id: str
    room: str
    identity: str
    user_id: str | None
    npc_id: str | None
    persona: dict[str, Any] | None
    voice: str | None
    language: str | None
    llm_model: str | None
    context: dict[str, Any] | None
    status: str
    created_at: str
    started_at: str | None
    ended_at: str | None
    audio_seconds: int | None

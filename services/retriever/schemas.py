"""Pydantic request/response schemas for the retriever HTTP surface.

Mirrors `specs/002-helper-guide-npc/contracts/retrieve.openapi.yaml`.
The SQLAlchemy ORM lives in `models.py`; these are the over-the-wire
types. Keeping them in a dedicated module (a) prevents accidental leak
of ORM attributes into the HTTP body and (b) gives the OpenAPI
generator clean shapes to serialize.

`npc_id` and `language` are declared as plain strings with defaults
rather than `Literal[...]` so that unsupported values raise our typed
`UnsupportedNpc` / `UnsupportedLanguage` with the Error envelope —
Pydantic's default for a `Literal` mismatch is HTTP 422, which would
violate the contract (must be 400 + error code per the OpenAPI spec).
The handler validates these explicitly.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryTurn(BaseModel):
    """One prior-turn entry passed to the rewriter."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)


class RetrieveRequest(BaseModel):
    """POST /retrieve request body."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: uuid.UUID
    session_id: uuid.UUID
    turn_id: str = Field(min_length=1, max_length=64)
    transcript: str = Field(min_length=1, max_length=4000)
    # Validated by the handler against a typed-error set (see module
    # docstring for why these aren't `Literal` fields).
    npc_id: str = Field(default="helper_guide", min_length=1, max_length=64)
    language: str = Field(default="ru", min_length=2, max_length=8)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)
    top_k: int = Field(default=5, ge=1, le=20)


class FusionComponentsOut(BaseModel):
    """Per-chunk decomposition of the fused score — for ops + ablation."""

    model_config = ConfigDict(extra="forbid")

    semantic: float
    lexical: float


class RetrievedChunkOut(BaseModel):
    """One chunk in the RetrieveResponse.chunks array."""

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    content: str
    score: float
    fusion_components: FusionComponentsOut
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTimings(BaseModel):
    """Per-stage wall-time in milliseconds.

    `total` is ALWAYS present. The stage fields are optional because
    the out-of-scope branch skips `embed` + `sql` (and the US2 refusal
    branch adds `pad`). The invariant `total == sum(non-total parts) ±
    1 ms` is enforced pre-insert by `traces.write_trace`.
    """

    model_config = ConfigDict(extra="forbid")

    rewrite: int | None = None
    embed: int | None = None
    sql: int | None = None
    pad: int | None = None
    total: int


class RetrieveResponse(BaseModel):
    """POST /retrieve 200 response body."""

    model_config = ConfigDict(extra="forbid")

    rewritten_query: str | None
    in_scope: bool
    chunks: list[RetrievedChunkOut]
    stage_timings_ms: StageTimings
    trace_id: uuid.UUID

"""Transcript write + read endpoints.

Two routes, two audiences:

    POST /internal/transcripts         agent-only, X-Internal-Token auth
    GET  /v1/sessions/{room}/transcripts  tenant-scoped, X-API-Key auth

The internal endpoint lives outside /v1/* because it is not part of the
public API contract — it's the private back-channel the agent worker
uses to log utterances. Exposing it over Caddy is fine; the shared
secret gate is what keeps it private.

Rows are immutable (no PATCH/DELETE), which makes retention policies
straightforward later.
"""

from __future__ import annotations

import datetime
import logging
import uuid

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import TenantIdentity
from db import get_db
from errors import APIError
from models import Session as SessionRow
from models import SessionTranscript
from rate_limit import enforce_tenant_rate_limit
from settings import settings

logger = logging.getLogger("project-800ms.api.transcripts")

# Internal endpoint has its own router so main.py can mount it under / (no
# /v1 prefix) while the read endpoint stays under /v1/.
internal_router = APIRouter(tags=["transcripts"])
v1_router = APIRouter(prefix="/v1", tags=["transcripts"])


class TranscriptWrite(BaseModel):
    """Payload the agent POSTs for each finalized utterance."""

    model_config = ConfigDict(extra="forbid")

    room: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., pattern="^(user|assistant)$")
    text: str = Field(..., min_length=1, max_length=8000)
    started_at: datetime.datetime | None = None
    ended_at: datetime.datetime | None = None


class TranscriptEntry(BaseModel):
    id: str
    role: str
    text: str
    started_at: str | None
    ended_at: str | None
    created_at: str


class TranscriptList(BaseModel):
    transcripts: list[TranscriptEntry]
    count: int


def _require_internal_token(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    """Guard for /internal/* routes.

    Empty config (agent_internal_token='') disables the endpoint entirely
    — surfacing a 503 instead of silently accepting an empty token.
    """
    if not settings.agent_internal_token:
        raise APIError(
            503,
            "agent_unavailable",
            "Internal transcript endpoint is not configured",
        )
    if not x_internal_token or x_internal_token != settings.agent_internal_token:
        raise APIError(401, "unauthenticated", "Invalid internal token")


@internal_router.post(
    "/internal/transcripts",
    status_code=201,
    summary="Ingest one transcript utterance (agent-only)",
    include_in_schema=False,
)
async def write_transcript(
    body: TranscriptWrite,
    _: None = Depends(_require_internal_token),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    # Resolve room -> session_id. We don't require the agent to know the
    # session UUID; room names are unique and the agent already has one.
    stmt = select(SessionRow).where(SessionRow.room == body.room)
    session = (await db.execute(stmt)).scalar_one_or_none()
    if session is None:
        # Unknown room — don't leak whether the session was deleted vs
        # never existed. Agent-side: log and move on.
        raise APIError(404, "not_found", "Session not found for this room")

    entry = SessionTranscript(
        session_id=session.id,
        role=body.role,
        text=body.text,
        started_at=body.started_at,
        ended_at=body.ended_at,
    )
    db.add(entry)
    await db.flush()
    await db.commit()

    logger.debug(
        "Transcript stored session=%s role=%s chars=%d",
        session.id,
        body.role,
        len(body.text),
    )
    return {"id": str(entry.id)}


@v1_router.get(
    "/sessions/{room}/transcripts",
    response_model=TranscriptList,
    summary="List transcripts for a session",
    description=(
        "Return all transcript entries for a session owned by the calling "
        "tenant, ordered by `created_at` ascending. Capped at 1000 entries "
        "per call — a single session shouldn't produce more, and this "
        "prevents a pathological query from scanning a huge table."
    ),
)
async def list_transcripts(
    request: Request,
    room: str,
    identity: TenantIdentity = Depends(enforce_tenant_rate_limit),
    db: AsyncSession = Depends(get_db),
) -> TranscriptList:
    # Tenant-scoped session lookup first. A 404 here (not 403) prevents
    # leaking whether another tenant holds this room.
    session_stmt = (
        select(SessionRow.id)
        .where(SessionRow.room == room)
        .where(SessionRow.tenant_id == uuid.UUID(identity.tenant_id))
    )
    session_id = (await db.execute(session_stmt)).scalar_one_or_none()
    if session_id is None:
        raise APIError(404, "not_found", f"Session {room!r} not found")

    stmt = (
        select(SessionTranscript)
        .where(SessionTranscript.session_id == session_id)
        .order_by(SessionTranscript.created_at.asc())
        .limit(1000)
    )
    rows = (await db.execute(stmt)).scalars().all()

    entries = [
        TranscriptEntry(
            id=str(r.id),
            role=r.role,
            text=r.text,
            started_at=r.started_at.isoformat() if r.started_at else None,
            ended_at=r.ended_at.isoformat() if r.ended_at else None,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
    return TranscriptList(transcripts=entries, count=len(entries))

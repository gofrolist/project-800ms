"""POST /v1/sessions + GET /v1/sessions/{room}.

Replaces the legacy unauthenticated POST /sessions. Every session is now
owned by the authenticated tenant: we insert a Session row (so we can
invoice, audit, and later look up transcripts by tenant) and dispatch the
agent with the full persona/voice/language payload so the pipeline can be
configured per-NPC.

Dispatch is best-effort synchronous for now — if the agent is down we roll
back the DB row and surface 503 agent_unavailable. This keeps sessions.room
from drifting out of sync with what's actually running in LiveKit.
"""

from __future__ import annotations

import datetime
import logging
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from livekit import api as lkapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import TenantIdentity
from db import get_db
from errors import APIError
from models import Session as SessionRow
from observability import room_var
from rate_limit import enforce_tenant_rate_limit
from schemas import CreateSessionRequest, CreateSessionResponse, SessionDetails
from settings import settings

logger = logging.getLogger("project-800ms.api.sessions")

router = APIRouter(prefix="/v1", tags=["sessions"])


# Shared response bodies for the error envelopes the /v1/sessions routes
# can emit. Keeping them in one place means the OpenAPI spec stays in
# sync with the actual error envelope shape.
_ERROR_ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "request_id": {"type": "string"},
            },
            "required": ["code", "message", "request_id"],
        }
    },
}

_COMMON_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {
        "description": "Missing or malformed X-API-Key header.",
        "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
    },
    403: {
        "description": "Key revoked or tenant suspended.",
        "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
    },
    422: {
        "description": "Request body failed validation.",
        "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
    },
    429: {
        "description": "Rate limit exceeded for this tenant.",
        "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
    },
}


def _mint_caller_token(room: str, identity: str) -> str:
    """Short-TTL LiveKit JWT for the caller's browser.

    Scoped to one room — a leaked token can't join anything else. TTL
    mirrors settings.session_ttl_seconds (default 15m), long enough for
    the handshake and then LiveKit's session lifecycle takes over.
    """
    return (
        lkapi.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(
            lkapi.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(datetime.timedelta(seconds=settings.session_ttl_seconds))
        .to_jwt()
    )


async def _dispatch_agent(room: str, body: CreateSessionRequest) -> None:
    """Tell the agent worker to spawn a pipeline for this room.

    Extra fields (persona, voice, language, llm_model, context) are sent
    today but the agent ignores anything beyond `room` until the Phase 1e
    pipeline changes land. Forward-compat safe — aiohttp's json.get drops
    unknown keys silently.
    """
    payload: dict[str, object] = {"room": room}
    if body.user_id is not None:
        payload["user_id"] = body.user_id
    if body.npc_id is not None:
        payload["npc_id"] = body.npc_id
    if body.persona is not None:
        payload["persona"] = body.persona
    if body.voice is not None:
        payload["voice"] = body.voice
    if body.language is not None:
        payload["language"] = body.language
    if body.llm_model is not None:
        payload["llm_model"] = body.llm_model
    if body.context is not None:
        payload["context"] = body.context
    if body.tts_engine is not None:
        payload["tts_engine"] = body.tts_engine

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                f"{settings.agent_dispatch_url}/dispatch",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Agent dispatch failed for room=%s: %s", room, exc)
            raise APIError(503, "agent_unavailable", "Agent is currently unavailable") from exc


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
    summary="Open a voice session",
    description=(
        "Create a new LiveKit room, dispatch the voice agent into it, and "
        "return the credentials the client needs to join.\n\n"
        "The caller connects to `url` over WebRTC using `token` and "
        "`identity`. Persona and voice fields are forwarded to the agent "
        "so the LLM + TTS can be tuned per-NPC."
    ),
    responses={
        503: {
            "description": "Agent dispatcher is unavailable; the session row has been rolled back.",
            "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
        },
        **_COMMON_ERROR_RESPONSES,
    },
)
async def create_session(
    request: Request,
    body: CreateSessionRequest,
    identity: TenantIdentity = Depends(enforce_tenant_rate_limit),
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    # Cross-field validation done imperatively in the route rather than as
    # a Pydantic field_validator. See CreateSessionRequest.
    # validate_engine_voice_combo for the rationale — the Pydantic →
    # RequestValidationError path was surfacing as 500 under this
    # project's BaseHTTPMiddleware stack; HTTPException(422) routes
    # cleanly through the registered http_exception_handler.
    try:
        body.validate_engine_voice_combo()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    room = f"room-{uuid.uuid4().hex[:8]}"
    room_var.set(room)
    caller_identity = body.user_id or f"user-{uuid.uuid4().hex[:8]}"

    session = SessionRow(
        tenant_id=uuid.UUID(identity.tenant_id),
        api_key_id=uuid.UUID(identity.api_key_id),
        room=room,
        identity=caller_identity,
        user_id=body.user_id,
        npc_id=body.npc_id,
        persona=body.persona,
        voice=body.voice,
        language=body.language,
        llm_model=body.llm_model,
        context=body.context,
        status="pending",
    )
    db.add(session)
    await db.flush()

    try:
        await _dispatch_agent(room, body)
    except APIError:
        await db.rollback()
        raise

    session.status = "active"
    session.started_at = datetime.datetime.now(datetime.UTC)
    await db.flush()
    await db.commit()

    token = _mint_caller_token(room, caller_identity)

    logger.info(
        "Session created tenant=%s key=%s room=%s npc=%s",
        identity.tenant_slug,
        identity.key_prefix,
        room,
        body.npc_id or "-",
    )

    return CreateSessionResponse(
        session_id=str(session.id),
        room=room,
        identity=caller_identity,
        token=token,
        url=settings.livekit_public_url,
    )


@router.get(
    "/sessions/{room}",
    response_model=SessionDetails,
    summary="Fetch session details",
    description=(
        "Return the current state of a session owned by the calling tenant. "
        "Returns **404 not_found** for sessions owned by other tenants — "
        "existence is deliberately not distinguishable across tenants."
    ),
    responses={
        404: {
            "description": "No session with this room name for this tenant.",
            "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
        },
        **_COMMON_ERROR_RESPONSES,
    },
)
async def get_session(
    request: Request,
    room: str,
    identity: TenantIdentity = Depends(enforce_tenant_rate_limit),
    db: AsyncSession = Depends(get_db),
) -> SessionDetails:
    room_var.set(room)
    stmt = (
        select(SessionRow)
        .where(SessionRow.room == room)
        .where(SessionRow.tenant_id == uuid.UUID(identity.tenant_id))
    )
    session = (await db.execute(stmt)).scalar_one_or_none()
    if session is None:
        # 404 whether the row exists under a different tenant or not at all —
        # don't leak existence across tenants.
        raise APIError(404, "not_found", f"Session {room!r} not found")

    return SessionDetails(
        session_id=str(session.id),
        room=session.room,
        identity=session.identity,
        user_id=session.user_id,
        npc_id=session.npc_id,
        persona=session.persona,
        voice=session.voice,
        language=session.language,
        llm_model=session.llm_model,
        context=session.context,
        status=session.status,
        created_at=session.created_at.isoformat(),
        started_at=session.started_at.isoformat() if session.started_at else None,
        ended_at=session.ended_at.isoformat() if session.ended_at else None,
        audio_seconds=session.audio_seconds,
    )

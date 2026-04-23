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

import asyncio
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


async def _delete_livekit_room(room: str) -> None:
    """Best-effort tear down of the LiveKit room.

    Calls LiveKit's ``RoomService.delete_room``, which is idempotent
    server-side (deleting a non-existent room is a no-op) and force-kicks
    every participant — including the agent bot, which then completes
    its pipeline shutdown via the same on_participant_left path the
    browser-close case uses.

    Errors are logged and swallowed intentionally. Rationale:
    - The primary teardown path is the agent's own on_participant_left →
      task.cancel, fired whenever the caller disconnects. DELETE is a
      belt-and-braces user-initiated signal and an operator tool.
    - If LiveKit's API is transiently unavailable, we still want to mark
      the session ``ended`` in the DB so billing, audit, and idempotency
      are consistent. The room will fall out on LiveKit's own
      ``empty_timeout`` once the caller's browser disconnects.
    - Returning 503 on LiveKit flakes would make DELETE un-idempotent and
      block the DB-side close the caller actually cares about.

    Timeout: 5 seconds. The LiveKit SDK's default is 60s (aiohttp default)
    which would stall the DELETE handler far longer than a best-effort
    cleanup should ever take. A hanging LiveKit should fall through to
    the DB-side close quickly; any actual teardown that needs >5s is
    pathological and the client can retry.
    """
    # Only narrow the catch below to the exceptions we actually expect from
    # LiveKit/aiohttp. ValueError (empty url/key), protobuf TypeError, or
    # asyncio.InvalidStateError indicate programmer errors that should
    # surface rather than be silently logged as a transient failure.
    import aiohttp  # noqa: PLC0415

    # Server-side URL takes precedence; fall back to the public URL for
    # deploys where the two are the same (prod behind a routable
    # hostname). See settings.livekit_url docstring for the split.
    server_url = settings.livekit_url or settings.livekit_public_url
    try:
        async with lkapi.LiveKitAPI(
            server_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as lk:
            await lk.room.delete_room(lkapi.DeleteRoomRequest(room=room))
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        # Network-level transient errors: log and continue. DB-side close
        # still commits; LiveKit will reclaim the room via empty_timeout
        # once the caller's browser disconnects.
        logger.warning(
            "LiveKit delete_room failed for room=%s: %s. "
            "Session will still be marked ended in the database.",
            room,
            exc,
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

    return _session_to_details(session)


def _session_to_details(session: SessionRow) -> SessionDetails:
    """Project a Session row to its SessionDetails API shape.

    Shared between GET /v1/sessions/{room} and DELETE /v1/sessions/{room}
    so adding a field to SessionDetails can't leave one endpoint out of
    sync with the other.
    """
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


# Terminal session statuses for which DELETE is a no-op short-circuit.
# ``ended`` is the normal close path; ``failed`` means the pipeline or
# dispatch errored and the row already carries its final state — DELETE
# must preserve it rather than overwrite with ``ended``, or we'd lose
# diagnostic signal for incident triage.
_SESSION_TERMINAL_STATUSES: frozenset[str] = frozenset({"ended", "failed"})


@router.delete(
    "/sessions/{room}",
    response_model=SessionDetails,
    summary="End a voice session",
    description=(
        "Force-close a session: tears down the LiveKit room (kicking the "
        "caller and the agent bot) and marks the session ``ended`` in the "
        "database.\n\n"
        "Idempotent: calling DELETE on a session that is already ``ended`` "
        "or ``failed`` returns 200 with the existing details and does not "
        "re-tear the already-deleted LiveKit room. The terminal ``failed`` "
        "status is preserved (not overwritten with ``ended``) so incident "
        "diagnostic state survives retries.\n\n"
        "Returns **409 conflict** for sessions still in ``pending`` — "
        "``started_at`` has not been populated yet (POST is mid-flight), so "
        "closing now would produce a row invisible to usage reports. "
        "Client should retry shortly.\n\n"
        "Returns **404 not_found** for sessions owned by other tenants or "
        "that do not exist — existence is deliberately not distinguishable "
        "across tenants. Belt-and-braces on top of the agent's own "
        "participant-left auto-teardown; this endpoint lets the UI's "
        '"hang up" button release resources immediately rather than '
        "waiting on LiveKit's ``empty_timeout``."
    ),
    responses={
        404: {
            "description": "No session with this room name for this tenant.",
            "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
        },
        409: {
            "description": "Session is still pending (not yet active); cannot close yet.",
            "content": {"application/json": {"schema": _ERROR_ENVELOPE_SCHEMA}},
        },
        **_COMMON_ERROR_RESPONSES,
    },
)
async def delete_session(
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
        # Same cross-tenant opacity rule as GET: don't leak existence.
        raise APIError(404, "not_found", f"Session {room!r} not found")

    # DELETE on status='pending' would produce an ended row with
    # started_at=NULL, which usage.py's ``WHERE started_at >= start``
    # filter excludes entirely — the call vanishes from reports. Tell
    # the client to retry once POST commits (status transitions
    # pending → active in create_session before returning 201, so this
    # window is narrow but real under high concurrency).
    if session.status == "pending":
        raise APIError(
            409,
            "conflict",
            f"Session {room!r} is still pending; retry once it is active",
        )

    # Idempotent short-circuit: if the session is already at a terminal
    # status (ended or failed), skip the LiveKit call (the room is gone)
    # and the DB write (nothing to change). ``failed`` is preserved as-is
    # so diagnostic state survives — we refuse to overwrite it with
    # ``ended`` just because a client sent DELETE.
    if session.status in _SESSION_TERMINAL_STATUSES:
        logger.info(
            "DELETE on already-closed session tenant=%s key=%s room=%s status=%s",
            identity.tenant_slug,
            identity.key_prefix,
            room,
            session.status,
        )
        return _session_to_details(session)

    # LiveKit first — actually kick the agent + caller out of the
    # room. The agent's on_participant_left auto-teardown then fires
    # and the pipeline drains. Any LiveKit API error is logged and
    # swallowed so the DB close below still runs (see
    # _delete_livekit_room docstring).
    await _delete_livekit_room(room)

    # Conditional UPDATE guards against concurrent DELETE races: if two
    # requests both pass the status check above, only one row update
    # lands ``status='ended'`` in the row; the second finds 0 affected
    # rows and returns the already-closed state below. This keeps
    # ``ended_at`` stable across retries (matters for billing joins).
    now = datetime.datetime.now(datetime.UTC)
    audio_seconds = (
        max(int((now - session.started_at).total_seconds()), 0)
        if session.started_at is not None
        else None
    )
    update_stmt = (
        SessionRow.__table__.update()
        .where(SessionRow.id == session.id)
        .where(SessionRow.status == session.status)
        .values(status="ended", ended_at=now, audio_seconds=audio_seconds)
    )
    result = await db.execute(update_stmt)
    await db.commit()

    if result.rowcount == 0:
        # Another concurrent DELETE (or the room_finished webhook)
        # beat us to the update. Re-read so the response reflects the
        # winning write's timestamps rather than our local mutations.
        await db.refresh(session)
        logger.info(
            "DELETE raced with concurrent close tenant=%s key=%s room=%s winner_status=%s",
            identity.tenant_slug,
            identity.key_prefix,
            room,
            session.status,
        )
        return _session_to_details(session)

    # Reflect the UPDATE in the in-memory session object so the response
    # body uses the values we actually wrote.
    session.status = "ended"
    session.ended_at = now
    session.audio_seconds = audio_seconds

    logger.info(
        "Session ended tenant=%s key=%s room=%s audio_seconds=%s",
        identity.tenant_slug,
        identity.key_prefix,
        room,
        audio_seconds,
    )
    return _session_to_details(session)

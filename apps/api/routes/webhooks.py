"""POST /v1/livekit-webhook — inbound events from the LiveKit SFU.

LiveKit fires events at a configured URL (see infra/livekit.prod.yaml).
Each request carries an `Authorization` header with a JWT signed using
the matching api-secret; the JWT's `sha` claim is the SHA-256 of the
request body so the SDK's WebhookReceiver both authenticates the caller
AND verifies body integrity.

We act on three events today:
    room_started       — update sessions.status='active' + started_at
    room_finished      — update sessions.status='ended'  + ended_at + audio_seconds
    participant_joined — defensive remove_participant if the joiner's
                         session is already at a terminal status, closing
                         the DELETE-before-join ghost-room race

Everything else is acknowledged with 204 (LiveKit retries on non-2xx).
Unknown rooms are also 204 — someone could be using the same LiveKit
instance for a test room that predates the sessions table, and we don't
want to wedge their retry loop.

The endpoint is not in /v1/* from a client perspective — it's
server-to-server from LiveKit to us. It lives under /v1/ for route
organization only; there's no X-API-Key, no tenant scoping, and
certainly no browser-origin check.
"""

from __future__ import annotations

import datetime
import logging

import aiohttp
import jwt
from fastapi import APIRouter, Depends, Header, Request, Response
from livekit import api as lkapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from errors import APIError
from models import Session as SessionRow
from rate_limit import enforce_webhook_ip_rate_limit
from settings import settings

# Statuses that mean the session is over. A participant joining a room
# whose session is at any of these states is a ghost-join — the caller's
# JWT is still valid but the pipeline has been torn down, so we kick the
# participant immediately rather than let them sit in an empty room for
# the token's remaining 15 minutes. Mirrors the terminal-status set in
# apps/api/routes/sessions.py.
_TERMINAL_SESSION_STATUSES: frozenset[str] = frozenset({"ended", "failed"})

# Agent bot identity prefix from services/agent/main.py AGENT_IDENTITY.
# participant_joined fires for both the caller AND the agent; the agent
# joining a room whose session is 'active' (i.e. the normal lifecycle)
# must NOT be kicked, obviously, but the agent joining a room whose
# session already flipped to 'ended' (because DELETE raced the agent
# dispatch) also must not be kicked — the agent's own on_participant_left
# hook fires as soon as the caller leaves and takes care of teardown.
# Use the identity prefix to identify the agent and skip the ghost-kick.
_AGENT_IDENTITY_PREFIX = "agent-bot-"

logger = logging.getLogger("project-800ms.api.webhooks")

router = APIRouter(prefix="/v1", tags=["webhooks"])

# Single receiver — TokenVerifier is stateless so sharing the instance
# across requests is safe. The LiveKit server uses `livekit_api_secret` to
# sign; we verify with the same key.
_receiver = lkapi.WebhookReceiver(
    lkapi.TokenVerifier(settings.livekit_api_key, settings.livekit_api_secret)
)

# Pre-built JWT deliberately signed with a bogus key so _receiver.receive()
# parses the header + payload, runs HMAC verification, and raises. This is
# the same computational path as a real bad-signature request, so calling
# it on the missing-header branch gives the two 401 responses matching
# latency profiles — closing the timing side-channel that a unified error
# string alone cannot close.
_TIMING_PARITY_DUMMY_JWT = jwt.encode(
    {"iss": "timing-parity-placeholder", "exp": 9_999_999_999, "sha256": ""},
    "x" * 32,  # deliberately wrong — verification will fail
    algorithm="HS256",
)


async def _remove_participant(room: str, identity: str) -> None:
    """Best-effort kick of a participant from a LiveKit room.

    Called when a participant_joined event fires for a session that is
    already terminal (ended / failed) — closes the DELETE-before-join
    race where the caller's JWT is still valid but the session is over.

    Errors are logged and swallowed (same policy as
    ``_delete_livekit_room`` in routes/sessions.py): a failure here
    leaves the ghost caller in a now-empty room for up to the JWT's
    15-minute TTL, which is annoying but not a correctness bug. The
    alternative — propagating 5xx to LiveKit — would cause the webhook
    to be retried, which is pointless since the participant is almost
    certainly still there. Retries make the race worse, not better.
    """
    server_url = settings.livekit_url or settings.livekit_public_url
    try:
        async with lkapi.LiveKitAPI(
            server_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as lk:
            await lk.room.remove_participant(
                lkapi.RoomParticipantIdentity(room=room, identity=identity)
            )
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        logger.warning(
            "LiveKit remove_participant failed for room=%s identity=%s: %s. "
            "Ghost participant will remain until JWT expires.",
            room,
            identity,
            exc,
        )


@router.post(
    "/livekit-webhook",
    status_code=204,
    summary="LiveKit webhook receiver",
    include_in_schema=False,
)
async def livekit_webhook(
    request: Request,
    authorization: str | None = Header(default=None),
    _: None = Depends(enforce_webhook_ip_rate_limit),
    db: AsyncSession = Depends(get_db),
) -> Response:
    # Read the body up-front so both 401 branches (missing header, bad
    # signature) do the same work. FastAPI caches the raw body, so the
    # later `event = _receiver.receive(raw_body, ...)` reuses the same
    # bytes without a second read.
    raw_body = (await request.body()).decode("utf-8")

    # One generic 401 for both branches — closes the lexical signal
    # (response body is identical) *and* the timing signal (both paths
    # run _receiver.receive so an attacker measuring response latency
    # can't distinguish "header not sent" from "signature bad").
    if not authorization:
        try:
            _receiver.receive(raw_body, _TIMING_PARITY_DUMMY_JWT)
        except Exception:
            # Expected — the dummy JWT is signed with a bogus key. The
            # purpose of the call is the crypto work, not the result.
            pass
        logger.warning("Webhook: missing Authorization header")
        raise APIError(401, "unauthenticated", "Webhook authentication failed")

    try:
        event = _receiver.receive(raw_body, authorization)
    except Exception as exc:  # JWT / signature / sha mismatch
        logger.warning("Webhook verification failed: %s", exc)
        raise APIError(401, "unauthenticated", "Webhook authentication failed") from exc

    room_name = event.room.name if event.room else ""
    if not room_name:
        # room_started / room_finished always carry a room; other events
        # we don't care about may not. 204 as a safe ack.
        return Response(status_code=204)

    stmt = select(SessionRow).where(SessionRow.room == room_name)
    session = (await db.execute(stmt)).scalar_one_or_none()
    if session is None:
        # Unknown room — LiveKit may be running rooms we didn't mint
        # (admin-created, other tenants on shared infra, etc.). Ack so
        # LiveKit stops retrying.
        logger.info("Webhook for unknown room=%s event=%s", room_name, event.event)
        return Response(status_code=204)

    now = datetime.datetime.now(datetime.UTC)

    if event.event == "room_started":
        if session.status == "pending":
            session.status = "active"
            session.started_at = now
        # If the session is already active/ended, room_started is a
        # retry or an out-of-order event — leave state alone.
        logger.info(
            "Webhook room_started room=%s tenant=%s",
            room_name,
            session.tenant_id,
        )

    elif event.event == "room_finished":
        # Idempotent: only transition into `ended` once. If we've already
        # ended this room, a replayed event is a no-op.
        if session.ended_at is None:
            session.ended_at = now
            session.status = "ended"
            if session.started_at is not None:
                delta = now - session.started_at
                session.audio_seconds = max(int(delta.total_seconds()), 0)
            else:
                # No room_started seen (event loss or webhook 429) — fall
                # back to event-carried room.creation_time for a best-effort
                # duration. If creation_time is also missing, audio_seconds
                # silently stays at 0, which is indistinguishable from a
                # legit zero-second session in downstream billing/analytics.
                # Log the fallback so this case is visible instead of silent.
                creation_ts = event.room.creation_time if event.room else 0
                if creation_ts:
                    session.audio_seconds = max(int(now.timestamp() - creation_ts), 0)
                    logger.warning(
                        "Webhook room_finished fallback room=%s tenant=%s: "
                        "no room_started observed; computed audio_seconds=%s "
                        "from event.room.creation_time",
                        room_name,
                        session.tenant_id,
                        session.audio_seconds,
                    )
                else:
                    logger.warning(
                        "Webhook room_finished incomplete room=%s tenant=%s: "
                        "no room_started observed and event.room.creation_time "
                        "missing; audio_seconds=0 may be inaccurate. Check for "
                        'dropped webhooks (see http_requests_total{status="429"} '
                        "for path=/v1/livekit-webhook).",
                        room_name,
                        session.tenant_id,
                    )
        logger.info(
            "Webhook room_finished room=%s tenant=%s audio_seconds=%s",
            room_name,
            session.tenant_id,
            session.audio_seconds,
        )

    elif event.event == "participant_joined":
        # Ghost-join defense: a participant joining a room whose session
        # is already at a terminal status means the DELETE /v1/sessions
        # call tore down the pipeline but the caller's JWT is still
        # valid (15min TTL). LiveKit auto-creates an empty room on
        # token-valid join — kick immediately so the caller sees a
        # disconnect instead of sitting alone in silence.
        #
        # Skip the agent bot's own join (identity prefix agent-bot-*):
        # during normal lifecycle the agent joins BEFORE the session
        # flips to ended, so we shouldn't land here, but during a
        # DELETE-vs-dispatch race the agent could arrive after the
        # status flip and we'd be kicking it from its own cleanup path.
        # Let the agent's on_participant_left auto-teardown handle it.
        if session.status in _TERMINAL_SESSION_STATUSES and event.participant is not None:
            identity = event.participant.identity
            if identity.startswith(_AGENT_IDENTITY_PREFIX):
                logger.info(
                    "Ghost-join defense skipped for agent identity=%s room=%s status=%s",
                    identity,
                    room_name,
                    session.status,
                )
            else:
                logger.info(
                    "Ghost-join detected: kicking participant identity=%s from "
                    "room=%s tenant=%s (session.status=%s)",
                    identity,
                    room_name,
                    session.tenant_id,
                    session.status,
                )
                await _remove_participant(room_name, identity)
        return Response(status_code=204)

    else:
        # participant_left, track_published, etc. — not used today. Ack
        # without DB writes.
        return Response(status_code=204)

    await db.commit()
    return Response(status_code=204)

"""POST /v1/livekit-webhook — inbound events from the LiveKit SFU.

LiveKit fires events at a configured URL (see infra/livekit.prod.yaml).
Each request carries an `Authorization` header with a JWT signed using
the matching api-secret; the JWT's `sha` claim is the SHA-256 of the
request body so the SDK's WebhookReceiver both authenticates the caller
AND verifies body integrity.

We only act on two events today:
    room_started   — update sessions.status='active' + started_at
    room_finished  — update sessions.status='ended'  + ended_at + audio_seconds

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

    else:
        # participant_joined, participant_left, track_published, etc. —
        # not used today. Ack without DB writes.
        return Response(status_code=204)

    await db.commit()
    return Response(status_code=204)

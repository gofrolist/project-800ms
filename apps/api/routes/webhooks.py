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

from fastapi import APIRouter, Header, Request, Response
from livekit import api as lkapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from errors import APIError
from fastapi import Depends
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
    # One generic 401 for both missing header and bad signature — don't
    # let attackers distinguish "header not sent" from "signature bad",
    # which would otherwise be a trivial probing primitive.
    if not authorization:
        logger.warning("Webhook: missing Authorization header")
        raise APIError(401, "unauthenticated", "Webhook authentication failed")

    raw_body = (await request.body()).decode("utf-8")
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
                # No room_started seen (event loss) — fall back to
                # event-carried room.creation_time for a best-effort
                # duration.
                creation_ts = event.room.creation_time if event.room else 0
                if creation_ts:
                    session.audio_seconds = max(int(now.timestamp() - creation_ts), 0)
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

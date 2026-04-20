"""LiveKit webhook endpoint tests.

Signs requests with the real livekit-api SDK so we exercise the exact
verification path prod uses. Requires the Postgres testcontainer — marked
slow.
"""

from __future__ import annotations

import datetime
import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from db import get_db
from main import app
from models import Session as SessionRow
from rate_limit import _reset_buckets_for_tests
from settings import settings

pytestmark = pytest.mark.slow


@pytest.fixture
def override_db(db_session):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    _reset_buckets_for_tests()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_db):
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


def _sign(body: str, *, secret: str | None = None) -> str:
    """Produce an Authorization JWT that LiveKit would send for `body`.

    The livekit-api WebhookReceiver expects the `sha256` claim to be the
    base64-encoded SHA-256 of the raw body (not hex). This matches
    LiveKit server's signing implementation.
    """
    import base64
    import hashlib

    import jwt

    digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()
    claims = {
        "iss": settings.livekit_api_key,
        "exp": int(datetime.datetime.now(datetime.UTC).timestamp()) + 300,
        "sha256": digest,
    }
    return jwt.encode(claims, secret or settings.livekit_api_secret, algorithm="HS256")


def _event_body(event_type: str, room_name: str, *, creation_time: int | None = None) -> str:
    """Build the JSON payload LiveKit would POST for this event."""
    room: dict = {"name": room_name, "sid": "RM_test"}
    if creation_time is not None:
        room["creationTime"] = creation_time
    payload = {
        "event": event_type,
        "room": room,
        "id": f"EV_{event_type}_{room_name}",
        "createdAt": int(datetime.datetime.now(datetime.UTC).timestamp()),
    }
    return json.dumps(payload)


async def _insert_session(db_session, tenant, api_key_id, *, room: str, status: str = "pending"):
    """Add a bare Session row we can watch the webhook transition."""
    row = SessionRow(
        tenant_id=tenant.id,
        api_key_id=api_key_id,
        room=room,
        identity="user-test",
        status=status,
    )
    db_session.add(row)
    await db_session.flush()
    return row


async def test_missing_auth_returns_401(client):
    r = await client.post("/v1/livekit-webhook", content="{}")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"
    # Unified client-facing message — server log still distinguishes, but
    # the wire response does not, so attackers can't probe the header
    # handling by diffing error strings.
    assert r.json()["error"]["message"] == "Webhook authentication failed"


async def test_401_identical_between_missing_header_and_bad_signature(client):
    """Both webhook 401 paths must return the same body — no probing primitive."""
    r_missing = await client.post("/v1/livekit-webhook", content="{}")
    r_bad_sig = await client.post(
        "/v1/livekit-webhook",
        content="{}",
        headers={"Authorization": "clearly.not.a.valid.jwt"},
    )
    assert r_missing.status_code == 401
    assert r_bad_sig.status_code == 401
    # Fields that depend on response ordering (timestamp, request_id) aren't
    # part of the API contract here — code + message are what the client
    # sees and what an attacker could use to distinguish branches.
    err_missing = r_missing.json()["error"]
    err_bad_sig = r_bad_sig.json()["error"]
    assert err_missing["code"] == err_bad_sig["code"] == "unauthenticated"
    assert err_missing["message"] == err_bad_sig["message"] == "Webhook authentication failed"


async def test_wrong_signature_rejected(client, seed_tenant, db_session):
    tenant, _ = seed_tenant

    from models import ApiKey

    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    await _insert_session(db_session, tenant, api_key.id, room="room-sig")

    body = _event_body("room_started", "room-sig")
    bad_auth = _sign(body, secret="x" * 32)  # wrong secret
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": bad_auth, "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 401


async def test_room_started_marks_active(client, db_session, seed_tenant):
    from models import ApiKey

    tenant, _ = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    session = await _insert_session(db_session, tenant, api_key.id, room="room-a")

    body = _event_body("room_started", "room-a")
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 204

    await db_session.refresh(session)
    assert session.status == "active"
    assert session.started_at is not None


async def test_room_finished_marks_ended_and_records_duration(client, db_session, seed_tenant):
    from models import ApiKey

    tenant, _ = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    # Start in 'active' with a started_at 30s ago so duration is deterministic.
    session = SessionRow(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room="room-b",
        identity="u",
        status="active",
        started_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=30),
    )
    db_session.add(session)
    await db_session.flush()

    body = _event_body("room_finished", "room-b")
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 204

    await db_session.refresh(session)
    assert session.status == "ended"
    assert session.ended_at is not None
    # ~30s, allow jitter for test wall-clock delay.
    assert 28 <= (session.audio_seconds or 0) <= 35


async def test_room_finished_is_idempotent(client, db_session, seed_tenant):
    """LiveKit retries on non-2xx; a replayed room_finished must not
    double-count or clobber the recorded duration."""

    from models import ApiKey

    tenant, _ = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    session = SessionRow(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room="room-c",
        identity="u",
        status="active",
        started_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=10),
    )
    db_session.add(session)
    await db_session.flush()

    body = _event_body("room_finished", "room-c")
    await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    await db_session.refresh(session)
    first_ended_at = session.ended_at
    first_audio = session.audio_seconds

    # Replay the same event — signature is still valid.
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 204

    await db_session.refresh(session)
    assert session.ended_at == first_ended_at
    assert session.audio_seconds == first_audio


async def test_unknown_room_is_silently_acked(client):
    body = _event_body("room_started", "room-does-not-exist")
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 204


async def test_unknown_event_type_is_ignored(client, db_session, seed_tenant):
    from models import ApiKey

    tenant, _ = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    session = await _insert_session(db_session, tenant, api_key.id, room="room-d")

    body = _event_body("participant_joined", "room-d")
    r = await client.post(
        "/v1/livekit-webhook",
        content=body,
        headers={"Authorization": _sign(body), "Content-Type": "application/webhook+json"},
    )
    assert r.status_code == 204

    await db_session.refresh(session)
    # status untouched — we don't act on participant_joined today.
    assert session.status == "pending"
    assert session.started_at is None

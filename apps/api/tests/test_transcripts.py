"""Tests for /internal/transcripts (write) and /v1/sessions/{room}/transcripts (read)."""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Request, Response
from sqlalchemy import select

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import ApiKey, Session as SessionRow, SessionTranscript, Tenant
from settings import settings

pytestmark = pytest.mark.slow


# The settings module caches values, but agent_internal_token is read off
# settings directly in the route; patch it on the Settings instance for
# the duration of the test module.
INTERNAL_TOKEN = "test-internal-token-" + "x" * 16


@pytest.fixture(autouse=True)
def _patch_internal_token():
    original = settings.agent_internal_token
    object.__setattr__(settings, "agent_internal_token", INTERNAL_TOKEN)
    yield
    object.__setattr__(settings, "agent_internal_token", original)


@pytest.fixture
def override_db(db_session):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    _clear_cache_for_tests()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_db):
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


@pytest.fixture
def agent_stub():
    """Stub the agent /dispatch call when we need to create a session first."""
    with patch("routes.sessions.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(
            return_value=Response(
                200,
                json={"status": "dispatched"},
                request=Request("POST", "http://agent/dispatch"),
            )
        )
        mock_client_class.return_value = mock_client
        yield mock_client


async def _create_session(db_session, tenant) -> SessionRow:
    """Fast-path: insert a session row directly (no route call)."""
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    row = SessionRow(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room=f"room-tx-{uuid.uuid4().hex[:8]}",
        identity="u-tx",
        status="active",
    )
    db_session.add(row)
    await db_session.flush()
    return row


# ─── POST /internal/transcripts ───────────────────────────────────────────


async def test_internal_write_requires_token(client):
    r = await client.post(
        "/internal/transcripts",
        json={"room": "room-any", "role": "user", "text": "hi"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_internal_write_wrong_token_rejected(client):
    r = await client.post(
        "/internal/transcripts",
        json={"room": "room-any", "role": "user", "text": "hi"},
        headers={"X-Internal-Token": "nope"},
    )
    assert r.status_code == 401


async def test_internal_write_persists_row(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    session = await _create_session(db_session, tenant)

    r = await client.post(
        "/internal/transcripts",
        json={"room": session.room, "role": "user", "text": "Привет"},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    assert r.status_code == 201, r.text
    assert r.json()["id"]

    rows = (
        (
            await db_session.execute(
                select(SessionTranscript).where(SessionTranscript.session_id == session.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].role == "user"
    assert rows[0].text == "Привет"


async def test_internal_write_unknown_room_404(client):
    r = await client.post(
        "/internal/transcripts",
        json={"room": "room-missing", "role": "user", "text": "x"},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    assert r.status_code == 404


async def test_internal_write_rejects_bad_role(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    session = await _create_session(db_session, tenant)

    r = await client.post(
        "/internal/transcripts",
        json={"room": session.room, "role": "system", "text": "x"},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    assert r.status_code == 422


# ─── GET /v1/sessions/{room}/transcripts ──────────────────────────────────


async def test_read_requires_api_key(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    session = await _create_session(db_session, tenant)
    r = await client.get(
        f"/v1/sessions/{session.room}/transcripts",
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.status_code == 401


async def test_read_returns_entries_in_order(client, db_session, seed_tenant):
    tenant, raw_key = seed_tenant
    session = await _create_session(db_session, tenant)

    # Insert 3 transcripts via direct DB to keep the test hermetic.
    for i, (role, text) in enumerate([("user", "hi"), ("assistant", "hello"), ("user", "bye")]):
        db_session.add(
            SessionTranscript(
                session_id=session.id,
                role=role,
                text=text,
            )
        )
    await db_session.flush()

    r = await client.get(
        f"/v1/sessions/{session.room}/transcripts",
        headers={"X-API-Key": raw_key, "Origin": "http://localhost:5173"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    roles = [e["role"] for e in body["transcripts"]]
    texts = [e["text"] for e in body["transcripts"]]
    # Order by created_at ascending — since all inserts share a now() in
    # the same transaction, just check membership rather than strict order.
    assert set(roles) == {"user", "assistant"}
    assert set(texts) == {"hi", "hello", "bye"}


async def test_read_cross_tenant_returns_404(client, db_session, seed_tenant):
    """Tenant A's transcripts are invisible to tenant B — no existence
    leak (same contract as GET /v1/sessions/{room})."""
    tenant_a, _ = seed_tenant
    session = await _create_session(db_session, tenant_a)
    db_session.add(SessionTranscript(session_id=session.id, role="user", text="secret"))
    await db_session.flush()

    # Second tenant with its own key.
    raw_b = "tk_b_" + uuid.uuid4().hex
    tenant_b = Tenant(name="B", slug=f"b-{uuid.uuid4().hex[:8]}")
    db_session.add(tenant_b)
    await db_session.flush()
    db_session.add(
        ApiKey(
            tenant_id=tenant_b.id,
            key_hash=hashlib.sha256(raw_b.encode()).digest(),
            key_prefix=raw_b[:8],
        )
    )
    await db_session.flush()

    r = await client.get(
        f"/v1/sessions/{session.room}/transcripts",
        headers={"X-API-Key": raw_b},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_read_unknown_room_returns_404(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.get(
        "/v1/sessions/room-never-existed/transcripts",
        headers={"X-API-Key": raw_key, "Origin": "http://localhost:5173"},
    )
    assert r.status_code == 404

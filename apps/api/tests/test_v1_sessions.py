"""POST /v1/sessions + GET /v1/sessions/{room} tests.

Uses the real app (so middleware, error envelope, auth all match prod)
with two overrides:
    - get_db is redirected to the savepointed db_session fixture
    - httpx.AsyncClient.post is patched to stub the agent dispatch call

Requires the Postgres testcontainer — marked slow.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Request, Response
from sqlalchemy import select

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import Session as SessionRow

pytestmark = pytest.mark.slow


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
    """Patch httpx.AsyncClient.post so /dispatch always succeeds."""
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


@pytest.fixture
def agent_down():
    """Patch httpx.AsyncClient.post to raise HTTP errors (simulate agent down)."""
    import httpx

    with patch("routes.sessions.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client_class.return_value = mock_client
        yield mock_client


async def test_create_session_requires_auth(client):
    r = await client.post("/v1/sessions", json={})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_create_session_empty_body_works(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body.keys()) == {"session_id", "room", "identity", "token", "url"}
    assert body["room"].startswith("room-")
    assert body["identity"].startswith("user-")
    assert body["token"]  # JWT is non-empty
    assert agent_stub.post.await_count == 1


async def test_create_session_persists_row(client, db_session, seed_tenant, agent_stub):
    tenant, raw_key = seed_tenant
    payload: dict[str, Any] = {
        "user_id": "player-42",
        "npc_id": "merchant",
        "persona": {"backstory": "grumpy shopkeeper"},
        "voice": "ru_RU-denis-medium",
        "language": "ru",
        "llm_model": "llama-3.3-70b-versatile",
        "context": {"inventory": ["sword", "shield"]},
    }
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json=payload)
    assert r.status_code == 201, r.json()

    row = (
        await db_session.execute(select(SessionRow).where(SessionRow.room == r.json()["room"]))
    ).scalar_one()
    assert row.user_id == "player-42"
    assert row.npc_id == "merchant"
    assert row.persona == {"backstory": "grumpy shopkeeper"}
    assert row.voice == "ru_RU-denis-medium"
    assert row.language == "ru"
    assert row.llm_model == "llama-3.3-70b-versatile"
    assert row.context == {"inventory": ["sword", "shield"]}
    assert row.status == "active"
    assert row.started_at is not None
    assert str(row.tenant_id) == str(tenant.id)
    # user_id becomes the caller identity when provided.
    assert row.identity == "player-42"


async def test_create_session_forwards_payload_to_agent(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    payload = {"npc_id": "merchant", "voice": "ru_RU-denis-medium"}
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json=payload)
    assert r.status_code == 201

    sent = agent_stub.post.await_args
    dispatched = sent.kwargs["json"]
    assert dispatched["npc_id"] == "merchant"
    assert dispatched["voice"] == "ru_RU-denis-medium"
    assert dispatched["room"].startswith("room-")
    # Fields not provided shouldn't appear at all — keeps the agent-side
    # contract tight.
    assert "persona" not in dispatched
    assert "context" not in dispatched


async def test_create_session_forwards_tts_engine_to_agent(client, seed_tenant, agent_stub):
    """Three-button demo contract: POST /v1/sessions with tts_engine flows
    through the dispatch payload so the agent can pick the right TTS
    backend per-session."""
    _tenant, raw_key = seed_tenant
    for engine in ("piper", "silero", "qwen3"):
        agent_stub.post.reset_mock()
        r = await client.post(
            "/v1/sessions",
            headers={"X-API-Key": raw_key},
            json={"tts_engine": engine},
        )
        assert r.status_code == 201, r.text
        dispatched = agent_stub.post.await_args.kwargs["json"]
        assert dispatched["tts_engine"] == engine


async def test_create_session_omits_tts_engine_when_unset(client, seed_tenant, agent_stub):
    """Empty body must NOT inject tts_engine — the agent's TTS_ENGINE env
    default has to win when the client is silent about engine choice."""
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 201
    dispatched = agent_stub.post.await_args.kwargs["json"]
    assert "tts_engine" not in dispatched


async def test_create_session_rejects_unknown_tts_engine(client, seed_tenant, agent_stub):
    """Pydantic Literal validation refuses unknown engine values BEFORE
    the dispatch call — keeps bogus strings from ever reaching the agent."""
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"tts_engine": "bogus"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert agent_stub.post.await_count == 0


async def test_create_session_rejects_unknown_fields(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"bogus_field": "value"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    # extra="forbid" must refuse the unknown field before we hit the agent.
    assert agent_stub.post.await_count == 0


async def test_create_session_agent_down_returns_503_and_rolls_back(
    client, db_session, seed_tenant, agent_down
):
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "agent_unavailable"

    # No session row should have been persisted — the insert was rolled back.
    rows = (await db_session.execute(select(SessionRow))).scalars().all()
    assert rows == []


async def test_get_session_returns_details(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"npc_id": "merchant"},
    )
    room = r.json()["room"]

    g = await client.get(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert g.status_code == 200
    body = g.json()
    assert body["room"] == room
    assert body["npc_id"] == "merchant"
    assert body["status"] == "active"


async def test_get_session_cross_tenant_returns_404(client, db_session, seed_tenant, agent_stub):
    """Tenant A cannot fetch tenant B's session — we return 404, not 403,
    to avoid leaking existence."""
    import hashlib
    import uuid as _uuid

    from models import ApiKey, Tenant

    _tenant_a, key_a = seed_tenant

    # Spin up tenant B with its own key.
    raw_b = "tk_b_" + _uuid.uuid4().hex
    tenant_b = Tenant(name="B", slug=f"b-{_uuid.uuid4().hex[:8]}")
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

    # Tenant A creates a session.
    r = await client.post("/v1/sessions", headers={"X-API-Key": key_a}, json={})
    room = r.json()["room"]

    # Tenant B cannot see it.
    g = await client.get(f"/v1/sessions/{room}", headers={"X-API-Key": raw_b})
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "not_found"


async def test_get_session_not_found(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    g = await client.get("/v1/sessions/room-doesnotexist", headers={"X-API-Key": raw_key})
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "not_found"


async def test_legacy_sessions_endpoint_removed(client):
    """Burn-the-boats: POST /sessions no longer exists.

    Starlette's router produces a plain {"detail": "Not Found"} for
    unmatched paths — that's fine. The envelope is guaranteed only for
    /v1/* responses that pass through our handlers.
    """
    r = await client.post("/sessions", json={})
    assert r.status_code == 404

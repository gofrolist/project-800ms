"""Per-tenant Origin allowlist enforcement.

The seed_tenant fixture sets `allowed_origins=["http://localhost:5173"]`,
so those tests naturally exercise the browser path. We add a second
fixture for tenants that opt out (empty allowed_origins) — the common
server-to-server integration pattern.

Agent dispatch is stubbed so these tests stay DB-only.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, Request, Response

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import ApiKey, Tenant

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


@pytest_asyncio.fixture
async def unrestricted_tenant(db_session):
    """Tenant with empty allowed_origins — no browser-origin check applies."""
    raw_key = "tk_unr_" + uuid.uuid4().hex
    tenant = Tenant(
        name="Unrestricted",
        slug=f"unr-{uuid.uuid4().hex[:8]}",
        allowed_origins=[],  # opt-out
    )
    db_session.add(tenant)
    await db_session.flush()

    db_session.add(
        ApiKey(
            tenant_id=tenant.id,
            key_hash=hashlib.sha256(raw_key.encode()).digest(),
            key_prefix=raw_key[:8],
        )
    )
    await db_session.flush()
    return tenant, raw_key


async def test_matching_origin_allowed(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={
            "X-API-Key": raw_key,
            "Origin": "http://localhost:5173",
        },
        json={},
    )
    assert r.status_code == 201


async def test_mismatched_origin_rejected(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={
            "X-API-Key": raw_key,
            "Origin": "https://attacker.example.com",
        },
        json={},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"
    assert "attacker.example.com" in r.json()["error"]["message"]
    # Agent must not have been dispatched — check fails before the insert.
    assert agent_stub.post.await_count == 0


async def test_no_origin_header_allowed(client, seed_tenant, agent_stub):
    """Server-to-server clients omit Origin — those must pass through."""
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 201


async def test_empty_allowlist_skips_check(client, unrestricted_tenant, agent_stub):
    """Tenants with no allowed_origins opt out — any Origin should work."""
    _tenant, raw_key = unrestricted_tenant
    r = await client.post(
        "/v1/sessions",
        headers={
            "X-API-Key": raw_key,
            "Origin": "https://random.example.com",
        },
        json={},
    )
    assert r.status_code == 201


async def test_origin_check_applies_to_get_too(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    # Create a session so GET has something to find.
    create = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key, "Origin": "http://localhost:5173"},
        json={},
    )
    room = create.json()["room"]

    bad = await client.get(
        f"/v1/sessions/{room}",
        headers={"X-API-Key": raw_key, "Origin": "https://attacker.example.com"},
    )
    assert bad.status_code == 403


async def test_origin_check_runs_after_auth(client, seed_tenant):
    """Missing key should win over origin — auth failures are the first signal."""
    r = await client.post(
        "/v1/sessions",
        headers={"Origin": "http://localhost:5173"},
        json={},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"

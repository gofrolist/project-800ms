"""Per-tenant token-bucket rate limit tests."""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Request, Response

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import ApiKey, Tenant
from rate_limit import _reset_buckets_for_tests

pytestmark = pytest.mark.slow


@pytest.fixture
def override_db(db_session):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    _clear_cache_for_tests()
    _reset_buckets_for_tests()
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


async def _new_tenant(db_session, *, rate: int) -> tuple[Tenant, str]:
    """Insert a tenant with a specific rate_limit_per_minute."""
    raw_key = "tk_" + uuid.uuid4().hex
    t = Tenant(
        name=f"rate-{rate}",
        slug=f"rate-{rate}-{uuid.uuid4().hex[:6]}",
        rate_limit_per_minute=rate,
        allowed_origins=[],
    )
    db_session.add(t)
    await db_session.flush()
    db_session.add(
        ApiKey(
            tenant_id=t.id,
            key_hash=hashlib.sha256(raw_key.encode()).digest(),
            key_prefix=raw_key[:8],
        )
    )
    await db_session.flush()
    return t, raw_key


async def test_budget_exhausted_returns_429(client, db_session, agent_stub):
    """A tenant with rate_limit_per_minute=3 should get 201 three times,
    then 429 on the fourth call."""
    _tenant, raw_key = await _new_tenant(db_session, rate=3)

    for _ in range(3):
        r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
        assert r.status_code == 201, r.text

    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["code"] == "rate_limited"
    assert "3/minute" in body["error"]["message"]


async def test_buckets_are_per_tenant(client, db_session, agent_stub):
    """Exhausting tenant A's bucket must not affect tenant B."""
    _t_a, key_a = await _new_tenant(db_session, rate=2)
    _t_b, key_b = await _new_tenant(db_session, rate=2)

    for _ in range(2):
        r = await client.post("/v1/sessions", headers={"X-API-Key": key_a}, json={})
        assert r.status_code == 201
    # A exhausted.
    r = await client.post("/v1/sessions", headers={"X-API-Key": key_a}, json={})
    assert r.status_code == 429
    # B untouched.
    r = await client.post("/v1/sessions", headers={"X-API-Key": key_b}, json={})
    assert r.status_code == 201


async def test_health_uses_ip_limit_not_tenant_bucket(client):
    """/health is unauthenticated — it must keep working under slowapi's
    IP-based 60/min rate regardless of tenant bucket state."""
    for _ in range(5):
        r = await client.get("/health")
        assert r.status_code == 200


async def test_admin_patch_changes_effective_rate(client, db_session, agent_stub, seed_tenant):
    """After PATCH /v1/admin/tenants/{slug}, the new rate applies on the
    next call that bypasses the auth cache."""
    from settings import settings

    # Enable admin.
    object.__setattr__(settings, "admin_api_key", "admin-test-" + "x" * 16)
    try:
        tenant, raw_key = seed_tenant
        # Drop the tenant's rate to 1/minute.
        r = await client.patch(
            f"/v1/admin/tenants/{tenant.slug}",
            headers={"X-Admin-Key": settings.admin_api_key},
            json={"rate_limit_per_minute": 1},
        )
        assert r.status_code == 200

        # The tenant-cache (auth.py _cache) still remembers the OLD rate.
        # Clear it so the next request re-reads the DB and sees rate=1.
        _clear_cache_for_tests()
        _reset_buckets_for_tests()

        r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
        assert r.status_code == 201

        r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
        assert r.status_code == 429
        assert "1/minute" in r.json()["error"]["message"]
    finally:
        object.__setattr__(settings, "admin_api_key", "")


# ─── Pure-unit coverage of the bucket mechanics ────────────────────────────


def test_token_bucket_consumes_and_refills():
    """Direct unit test of _TokenBucket math — no HTTP, no DB."""
    from rate_limit import _TokenBucket

    b = _TokenBucket.for_rate(60)
    assert b.capacity == 60.0
    # Capacity filled at construction.
    for _ in range(60):
        assert b.consume()
    # Empty now.
    assert not b.consume()
    # After simulated elapsed time, tokens refill.
    b.last_refill -= 2.0  # pretend 2 seconds passed
    assert b.consume()  # refill rate = 1 tok/sec × 2s = 2 tokens available


def test_token_bucket_never_exceeds_capacity():
    from rate_limit import _TokenBucket

    b = _TokenBucket.for_rate(10)
    # Jump the clock way forward — tokens should cap at capacity.
    b.last_refill -= 3600.0
    # Consuming once triggers refill; 10 tokens then drop to 9.
    assert b.consume()
    assert b.tokens <= 10.0

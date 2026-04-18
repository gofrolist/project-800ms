"""X-API-Key auth dependency tests.

Covers hash determinism, missing/malformed/unknown keys, revoked keys,
suspended tenants, cache hit behavior, and request.state population.
Requires a real DB (Postgres testcontainer).
"""

from __future__ import annotations

import datetime
import hashlib
import uuid

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from auth import TenantIdentity, _clear_cache_for_tests, get_current_tenant, hash_key
from db import get_db
from errors import install as install_errors
from models import ApiKey, Tenant
from request_id import RequestIdMiddleware

pytestmark = pytest.mark.slow


def test_hash_key_is_deterministic_and_32_bytes():
    h1 = hash_key("some-test-key")
    h2 = hash_key("some-test-key")
    assert h1 == h2
    assert len(h1) == 32
    assert h1 == hashlib.sha256(b"some-test-key").digest()


@pytest.fixture
def app(db_session):
    """Build a tiny app that mounts one protected endpoint.

    Uses a dependency override so every request reuses the same
    savepointed session as the test fixture — gives us transactional
    isolation without plumbing another engine.
    """

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    install_errors(app)

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    @app.get("/v1/whoami")
    async def whoami(identity: TenantIdentity = Depends(get_current_tenant)):
        return {
            "tenant_slug": identity.tenant_slug,
            "rate_limit_per_minute": identity.rate_limit_per_minute,
            "key_prefix": identity.key_prefix,
        }

    _clear_cache_for_tests()
    return app


@pytest.fixture
def client(app):
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_missing_header_returns_401_envelope(client):
    r = await client.get("/v1/whoami")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "unauthenticated"
    # Validation error from FastAPI's Header(...) required path. Message
    # shape depends on whether our handler or starlette's fires first.
    assert "request_id" in body["error"]


async def test_malformed_header_rejected(client):
    r = await client.get("/v1/whoami", headers={"X-API-Key": "short"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_unknown_key_rejected(client):
    r = await client.get("/v1/whoami", headers={"X-API-Key": "tk_" + "x" * 32})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"
    assert "Invalid API key" in r.json()["error"]["message"]


async def test_valid_key_returns_tenant(client, seed_tenant):
    tenant, raw_key = seed_tenant
    r = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_slug"] == tenant.slug
    assert body["rate_limit_per_minute"] == tenant.rate_limit_per_minute
    assert body["key_prefix"] == raw_key[:8]


async def test_revoked_key_rejected(client, db_session, seed_tenant):
    from sqlalchemy import select

    tenant, raw_key = seed_tenant
    _clear_cache_for_tests()  # ensure we hit the DB

    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    api_key.revoked_at = datetime.datetime.now(datetime.UTC)
    await db_session.flush()

    r = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"
    assert "revoked" in r.json()["error"]["message"].lower()


async def test_suspended_tenant_rejected(client, db_session, seed_tenant):
    tenant, raw_key = seed_tenant
    _clear_cache_for_tests()
    tenant.status = "suspended"
    await db_session.flush()

    r = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_cache_hit_skips_db(client, db_session, seed_tenant):
    """Second call for the same key should not hit the DB — we verify by
    revoking mid-test and observing the cached response is still served
    (briefly) within the cache TTL.
    """
    tenant, raw_key = seed_tenant

    # First call warms the cache.
    r1 = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r1.status_code == 200

    # Revoke directly via the session — cache should still serve success.
    from sqlalchemy import select

    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()
    api_key.revoked_at = datetime.datetime.now(datetime.UTC)
    await db_session.flush()

    r2 = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r2.status_code == 200  # cache hit — stale success within TTL

    # Clearing the cache forces a DB fetch and reveals the revocation.
    _clear_cache_for_tests()
    r3 = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r3.status_code == 403


async def test_request_state_populated(app, client, seed_tenant):
    """Verify request.state fields that the rate limiter depends on."""
    tenant, raw_key = seed_tenant
    captured = {}

    @app.middleware("http")
    async def _capture(request, call_next):
        response = await call_next(request)
        captured["tenant_id"] = getattr(request.state, "tenant_id", None)
        captured["rate"] = getattr(request.state, "tenant_rate_limit_per_minute", None)
        return response

    r = await client.get("/v1/whoami", headers={"X-API-Key": raw_key})
    assert r.status_code == 200
    assert captured["tenant_id"] == str(tenant.id)
    assert captured["rate"] == tenant.rate_limit_per_minute


async def test_two_different_keys_cached_independently(client, db_session):
    """A second tenant's key doesn't poison the first tenant's cache entry."""
    import hashlib

    from models import ApiKey

    raw_a = "tk_aaa_" + uuid.uuid4().hex
    raw_b = "tk_bbb_" + uuid.uuid4().hex

    t_a = Tenant(name="A", slug=f"a-{uuid.uuid4().hex[:6]}")
    t_b = Tenant(name="B", slug=f"b-{uuid.uuid4().hex[:6]}")
    db_session.add_all([t_a, t_b])
    await db_session.flush()

    db_session.add_all(
        [
            ApiKey(
                tenant_id=t_a.id,
                key_hash=hashlib.sha256(raw_a.encode()).digest(),
                key_prefix=raw_a[:8],
            ),
            ApiKey(
                tenant_id=t_b.id,
                key_hash=hashlib.sha256(raw_b.encode()).digest(),
                key_prefix=raw_b[:8],
            ),
        ]
    )
    await db_session.flush()
    _clear_cache_for_tests()

    r_a = await client.get("/v1/whoami", headers={"X-API-Key": raw_a})
    r_b = await client.get("/v1/whoami", headers={"X-API-Key": raw_b})
    assert r_a.json()["tenant_slug"] == t_a.slug
    assert r_b.json()["tenant_slug"] == t_b.slug

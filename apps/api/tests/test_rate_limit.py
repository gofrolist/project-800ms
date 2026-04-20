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
from settings import settings

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


# ─── IP-based limits (admin + webhook) ─────────────────────────────────────


@pytest.fixture
def enable_admin():
    """Temporarily configure an admin key so /v1/admin/* returns 401 (not 503)
    on wrong-key requests — needed for tests that probe auth ordering."""
    original = settings.admin_api_key
    object.__setattr__(settings, "admin_api_key", "admin-test-" + "x" * 32)
    yield
    object.__setattr__(settings, "admin_api_key", original)


async def test_admin_ip_rate_limit_fires_before_admin_key_check(client, enable_admin):
    """Bad X-Admin-Key requests still consume the admin-ip bucket.

    The router-level `enforce_admin_ip_rate_limit` must run before
    `_require_admin`, otherwise an attacker can brute-force the admin key
    at unlimited rate. Proof: send 60 requests with a wrong admin key
    (each returns 401 after the bucket check consumes a token), then the
    61st request from the same IP returns 429 — proving the rate limit is
    still running even though auth is failing.
    """
    # Sanity: test reasons about a 60/min ceiling — if ops widened it,
    # the 61st-request-is-429 assertion would flake. Pin it to the
    # tested value.
    original_rate = settings.admin_ip_rate_per_minute
    object.__setattr__(settings, "admin_ip_rate_per_minute", 60)
    _reset_buckets_for_tests()
    try:
        headers = {"X-Admin-Key": "not-the-real-key"}
        for i in range(60):
            r = await client.get("/v1/admin", headers=headers)
            assert r.status_code == 401, (
                f"request {i + 1}: expected 401 unauth, got {r.status_code}"
            )

        r = await client.get("/v1/admin", headers=headers)
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "rate_limited"
    finally:
        object.__setattr__(settings, "admin_ip_rate_per_minute", original_rate)


async def test_admin_ip_rate_limit_isolated_per_ip(client, enable_admin):
    """Two distinct X-Forwarded-For values must get independent buckets.

    Exhaust IP #1's budget via XFF, then send one request with XFF #2 —
    it must not inherit IP #1's exhausted state. Requires overriding
    `trusted_proxy_cidrs` to include the ASGI test client's loopback
    peer so `_real_ip` actually honours the XFF header in the test.
    """
    original_cidrs = settings.trusted_proxy_cidrs
    original_rate = settings.admin_ip_rate_per_minute
    object.__setattr__(settings, "trusted_proxy_cidrs", ["127.0.0.0/8"])
    object.__setattr__(settings, "admin_ip_rate_per_minute", 60)
    _reset_buckets_for_tests()
    try:
        headers_a = {"X-Admin-Key": "not-the-real-key", "X-Forwarded-For": "1.2.3.4"}
        headers_b = {"X-Admin-Key": "not-the-real-key", "X-Forwarded-For": "5.6.7.8"}

        for _ in range(60):
            r = await client.get("/v1/admin", headers=headers_a)
            assert r.status_code == 401
        r = await client.get("/v1/admin", headers=headers_a)
        assert r.status_code == 429  # IP #1 exhausted

        # IP #2 must still be able to hit the endpoint.
        r = await client.get("/v1/admin", headers=headers_b)
        assert r.status_code == 401  # auth fails, but NOT 429
    finally:
        object.__setattr__(settings, "trusted_proxy_cidrs", original_cidrs)
        object.__setattr__(settings, "admin_ip_rate_per_minute", original_rate)


async def test_webhook_ip_rate_limit_enforced(client):
    """Webhook route respects the IP-based 429 ceiling.

    Tune the webhook rate down to 5 via settings so we don't have to
    send 1001 requests. Bucket capacity is re-read in `_get_bucket` on
    each call, so a reset picks up the new cap immediately.
    """
    original = settings.webhook_ip_rate_per_minute
    object.__setattr__(settings, "webhook_ip_rate_per_minute", 5)
    _reset_buckets_for_tests()
    try:
        # Every call returns 401 (no valid auth) — but the rate-limit dep
        # runs first and consumes a token. After 5 attempts the bucket is
        # empty and the 6th call returns 429.
        for i in range(5):
            r = await client.post("/v1/livekit-webhook", content="{}")
            assert r.status_code == 401, f"request {i + 1}: got {r.status_code}"

        r = await client.post("/v1/livekit-webhook", content="{}")
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "rate_limited"
    finally:
        object.__setattr__(settings, "webhook_ip_rate_per_minute", original)


async def test_429_response_includes_retry_after_header(client):
    """Programmatic clients key their backoff on Retry-After. Pin it so a
    future refactor of the error handler doesn't drop the header."""
    original = settings.webhook_ip_rate_per_minute
    object.__setattr__(settings, "webhook_ip_rate_per_minute", 1)
    _reset_buckets_for_tests()
    try:
        # First call consumes the only token (still fails auth at 401).
        await client.post("/v1/livekit-webhook", content="{}")
        # Second call is 429 and must carry Retry-After.
        r = await client.post("/v1/livekit-webhook", content="{}")
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "60"
    finally:
        object.__setattr__(settings, "webhook_ip_rate_per_minute", original)


async def test_admin_ip_rate_limit_skipped_when_admin_disabled(client):
    """When settings.admin_api_key is empty, the admin surface is 503
    and the IP rate limit must not consume tokens. Prevents probing a
    disabled admin from evicting tenant buckets or punishing operators
    on a shared egress with 429 instead of the diagnostic 503."""
    # No `enable_admin` fixture here — admin_api_key stays empty.
    _reset_buckets_for_tests()
    # Send many more than the 60/min ceiling would allow if the limit
    # were active. All must return 503, none 429.
    for i in range(75):
        r = await client.get("/v1/admin", headers={"X-Admin-Key": "whatever"})
        assert r.status_code == 503, f"request {i + 1}: got {r.status_code}"

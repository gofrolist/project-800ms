"""/v1/admin/* tests."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from auth import _clear_cache_for_tests, hash_key
from db import get_db
from main import app
from models import ApiKey, Tenant
from rate_limit import _reset_buckets_for_tests
from settings import settings

pytestmark = pytest.mark.slow


ADMIN_KEY = "admin-test-" + "x" * 32


@pytest.fixture(autouse=True)
def _patch_admin_key():
    original = settings.admin_api_key
    object.__setattr__(settings, "admin_api_key", ADMIN_KEY)
    yield
    object.__setattr__(settings, "admin_api_key", original)


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


def _auth() -> dict[str, str]:
    return {"X-Admin-Key": ADMIN_KEY}


# ─── Auth ─────────────────────────────────────────────────────────────────


async def test_missing_admin_key_returns_401(client):
    r = await client.get("/v1/admin")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_wrong_admin_key_rejected(client):
    r = await client.get("/v1/admin", headers={"X-Admin-Key": "wrong"})
    assert r.status_code == 401


async def test_disabled_admin_api_returns_503(client):
    """Unconfigured admin_api_key (empty) disables the surface."""
    original = settings.admin_api_key
    object.__setattr__(settings, "admin_api_key", "")
    try:
        r = await client.get("/v1/admin")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "agent_unavailable"
    finally:
        object.__setattr__(settings, "admin_api_key", original)


async def test_probe_returns_ok(client):
    r = await client.get("/v1/admin", headers=_auth())
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ─── Tenants ──────────────────────────────────────────────────────────────


async def test_create_tenant(client, db_session):
    slug = f"acme-{uuid.uuid4().hex[:6]}"
    r = await client.post(
        "/v1/admin/tenants",
        headers=_auth(),
        json={
            "slug": slug,
            "name": "Acme Co",
            "rate_limit_per_minute": 120,
            "allowed_origins": ["https://acme.example.com"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["slug"] == slug
    assert body["name"] == "Acme Co"
    assert body["rate_limit_per_minute"] == 120
    assert body["allowed_origins"] == ["https://acme.example.com"]
    assert body["status"] == "active"

    row = (await db_session.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one()
    assert row.name == "Acme Co"


async def test_create_tenant_rejects_bad_slug(client):
    r = await client.post(
        "/v1/admin/tenants",
        headers=_auth(),
        json={"slug": "Bad Slug!", "name": "x"},
    )
    assert r.status_code == 422


async def test_create_tenant_conflict_on_duplicate_slug(client, seed_tenant):
    tenant, _ = seed_tenant
    r = await client.post(
        "/v1/admin/tenants",
        headers=_auth(),
        json={"slug": tenant.slug, "name": "dup"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"


async def test_list_tenants_returns_all(client, seed_tenant):
    r = await client.get("/v1/admin/tenants", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    slugs = {t["slug"] for t in body["tenants"]}
    assert seed_tenant[0].slug in slugs


async def test_patch_tenant(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    r = await client.patch(
        f"/v1/admin/tenants/{tenant.slug}",
        headers=_auth(),
        json={"rate_limit_per_minute": 5, "status": "suspended"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rate_limit_per_minute"] == 5
    assert body["status"] == "suspended"
    # Name untouched (partial update).
    assert body["name"] == tenant.name


async def test_patch_unknown_tenant_404(client):
    r = await client.patch(
        "/v1/admin/tenants/does-not-exist",
        headers=_auth(),
        json={"name": "x"},
    )
    assert r.status_code == 404


# ─── API keys ─────────────────────────────────────────────────────────────


async def test_issue_api_key_returns_raw_once(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    r = await client.post(
        f"/v1/admin/tenants/{tenant.slug}/api-keys",
        headers=_auth(),
        json={"label": "integration-ci"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["raw_key"].startswith("tk_")
    assert len(body["raw_key"]) > 32
    assert body["label"] == "integration-ci"
    assert body["key_prefix"] == body["raw_key"][:8]

    # Hash is what made it into the DB.
    row = (
        await db_session.execute(select(ApiKey).where(ApiKey.id == uuid.UUID(body["id"])))
    ).scalar_one()
    assert row.key_hash == hash_key(body["raw_key"])


async def test_issue_api_key_unknown_tenant_404(client):
    r = await client.post(
        "/v1/admin/tenants/does-not-exist/api-keys",
        headers=_auth(),
        json={"label": "x"},
    )
    assert r.status_code == 404


async def test_list_api_keys_is_metadata_only(client, seed_tenant):
    tenant, _ = seed_tenant
    r = await client.get(
        f"/v1/admin/tenants/{tenant.slug}/api-keys",
        headers=_auth(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    for entry in body["api_keys"]:
        # No raw key leak on list.
        assert "raw_key" not in entry
        assert len(entry["key_prefix"]) <= 8


async def test_revoke_api_key(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    # Use the seed tenant's existing key.
    existing = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    r = await client.post(
        f"/v1/admin/api-keys/{existing.id}/revoke",
        headers=_auth(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["revoked_at"] is not None


async def test_revoke_is_idempotent(client, db_session, seed_tenant):
    tenant, _ = seed_tenant
    existing = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    r1 = await client.post(f"/v1/admin/api-keys/{existing.id}/revoke", headers=_auth())
    first_ts = r1.json()["revoked_at"]

    r2 = await client.post(f"/v1/admin/api-keys/{existing.id}/revoke", headers=_auth())
    assert r2.status_code == 200
    assert r2.json()["revoked_at"] == first_ts  # unchanged


async def test_revoke_unknown_key_404(client):
    # Valid UUID shape, no row.
    r = await client.post(
        f"/v1/admin/api-keys/{uuid.uuid4()}/revoke",
        headers=_auth(),
    )
    assert r.status_code == 404


async def test_revoke_bad_uuid_404(client):
    r = await client.post(
        "/v1/admin/api-keys/not-a-uuid/revoke",
        headers=_auth(),
    )
    assert r.status_code == 404

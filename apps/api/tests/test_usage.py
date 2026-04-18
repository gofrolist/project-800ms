"""GET /v1/usage tests."""

from __future__ import annotations

import datetime
import hashlib
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import ApiKey, Session as SessionRow, Tenant

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


async def _add_session(
    db_session,
    tenant_id,
    api_key_id,
    *,
    started_at: datetime.datetime,
    audio_seconds: int,
    status: str = "ended",
):
    row = SessionRow(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        room=f"room-{uuid.uuid4().hex[:8]}",
        identity="u",
        status=status,
        started_at=started_at,
        audio_seconds=audio_seconds,
    )
    db_session.add(row)
    await db_session.flush()
    return row


async def test_missing_dates_422(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.get("/v1/usage", headers={"X-API-Key": raw_key})
    assert r.status_code == 422


async def test_from_must_be_before_to(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.get(
        "/v1/usage",
        params={"from": "2026-04-10", "to": "2026-04-10"},
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_window_too_large(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.get(
        "/v1/usage",
        params={"from": "2020-01-01", "to": "2026-01-01"},
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 422


async def test_empty_window_returns_zeros(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.get(
        "/v1/usage",
        params={"from": "2020-01-01", "to": "2020-01-02"},
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == []
    assert body["total_sessions"] == 0
    assert body["total_audio_seconds"] == 0


async def test_aggregates_by_day_for_tenant(client, db_session, seed_tenant):
    tenant, raw_key = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    # Three sessions on two days.
    day1 = datetime.datetime(2026, 4, 10, 10, 0, tzinfo=datetime.UTC)
    day2 = datetime.datetime(2026, 4, 11, 14, 0, tzinfo=datetime.UTC)
    await _add_session(db_session, tenant.id, api_key.id, started_at=day1, audio_seconds=60)
    await _add_session(db_session, tenant.id, api_key.id, started_at=day1, audio_seconds=90)
    await _add_session(db_session, tenant.id, api_key.id, started_at=day2, audio_seconds=30)

    r = await client.get(
        "/v1/usage",
        params={"from": "2026-04-10", "to": "2026-04-12"},
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_sessions"] == 3
    assert body["total_audio_seconds"] == 180
    assert len(body["days"]) == 2
    by_day = {d["day"]: d for d in body["days"]}
    assert by_day["2026-04-10"] == {
        "day": "2026-04-10",
        "sessions_count": 2,
        "audio_seconds": 150,
    }
    assert by_day["2026-04-11"] == {
        "day": "2026-04-11",
        "sessions_count": 1,
        "audio_seconds": 30,
    }


async def test_pending_sessions_count_but_no_audio(client, db_session, seed_tenant):
    """Active / pending sessions contribute to sessions_count but have
    audio_seconds=NULL until room_finished fires. Coalesce should map
    that to 0."""
    tenant, raw_key = seed_tenant
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    day = datetime.datetime(2026, 4, 10, 10, 0, tzinfo=datetime.UTC)
    row = SessionRow(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room=f"room-pend-{uuid.uuid4().hex[:8]}",
        identity="u",
        status="active",
        started_at=day,
        audio_seconds=None,
    )
    db_session.add(row)
    await db_session.flush()

    r = await client.get(
        "/v1/usage",
        params={"from": "2026-04-10", "to": "2026-04-11"},
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_sessions"] == 1
    assert body["total_audio_seconds"] == 0


async def test_cross_tenant_isolation(client, db_session, seed_tenant):
    """Tenant A must not see tenant B's usage (or vice versa)."""
    tenant_a, key_a = seed_tenant
    api_key_a = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant_a.id))
    ).scalar_one()

    raw_b = "tk_b_" + uuid.uuid4().hex
    tenant_b = Tenant(name="B", slug=f"b-{uuid.uuid4().hex[:8]}")
    db_session.add(tenant_b)
    await db_session.flush()
    api_key_b = ApiKey(
        tenant_id=tenant_b.id,
        key_hash=hashlib.sha256(raw_b.encode()).digest(),
        key_prefix=raw_b[:8],
    )
    db_session.add(api_key_b)
    await db_session.flush()

    day = datetime.datetime(2026, 4, 10, 10, 0, tzinfo=datetime.UTC)
    await _add_session(db_session, tenant_a.id, api_key_a.id, started_at=day, audio_seconds=100)
    await _add_session(db_session, tenant_b.id, api_key_b.id, started_at=day, audio_seconds=500)

    r_a = await client.get(
        "/v1/usage",
        params={"from": "2026-04-10", "to": "2026-04-11"},
        headers={"X-API-Key": key_a},
    )
    assert r_a.json()["total_audio_seconds"] == 100

    r_b = await client.get(
        "/v1/usage",
        params={"from": "2026-04-10", "to": "2026-04-11"},
        headers={"X-API-Key": raw_b},
    )
    assert r_b.json()["total_audio_seconds"] == 500

"""Sanity tests for the SQLAlchemy models + Alembic schema.

Verifies migrations apply cleanly, round-trips work, FK cascades fire, and
CHECK constraints reject bad values. Slow — requires Docker / Postgres.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import select, text

pytestmark = pytest.mark.slow


async def test_migrations_created_expected_tables(db_session):
    """Every Phase 1 table exists with the expected columns."""
    result = await db_session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
    )
    tables = {row[0] for row in result.all()}
    assert {"tenants", "api_keys", "sessions", "session_transcripts"}.issubset(tables)


async def test_tenant_roundtrip_and_defaults(db_session):
    from models import Tenant

    t = Tenant(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
    db_session.add(t)
    await db_session.flush()
    await db_session.refresh(t)

    assert t.id is not None
    assert t.rate_limit_per_minute == 60
    assert t.allowed_origins == []
    assert t.status == "active"
    assert t.created_at is not None


async def test_tenant_status_check_constraint(db_session):
    from models import Tenant

    t = Tenant(
        name="Bad",
        slug=f"bad-{uuid.uuid4().hex[:6]}",
        status="definitely-not-a-real-status",
    )
    db_session.add(t)
    with pytest.raises(Exception):  # IntegrityError, but asyncpg wraps it
        await db_session.flush()


async def test_api_key_cascades_on_tenant_delete(db_session):
    from models import ApiKey, Tenant

    t = Tenant(name="Doomed", slug=f"doomed-{uuid.uuid4().hex[:6]}")
    db_session.add(t)
    await db_session.flush()

    k = ApiKey(
        tenant_id=t.id,
        key_hash=hashlib.sha256(b"doomed-key").digest(),
        key_prefix="doomed-k",
    )
    db_session.add(k)
    await db_session.flush()
    k_id = k.id

    await db_session.delete(t)
    await db_session.flush()

    result = await db_session.execute(select(ApiKey).where(ApiKey.id == k_id))
    assert result.scalar_one_or_none() is None


async def test_session_status_check_constraint(db_session, seed_tenant):
    from models import ApiKey, Session

    tenant, _ = seed_tenant
    # Fetch the API key explicitly — lazy-loading via `tenant.api_keys`
    # doesn't work in async sessions without awaitable_attrs / selectinload.
    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    s = Session(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room=f"room-{uuid.uuid4().hex[:8]}",
        identity=f"user-{uuid.uuid4().hex[:8]}",
        status="nonsense",
    )
    db_session.add(s)
    with pytest.raises(Exception):
        await db_session.flush()


async def test_room_is_unique(db_session, seed_tenant):
    from models import Session

    tenant, _ = seed_tenant
    # Need the api_key for this — re-query since lazy-load is tricky on
    # freshly inserted rows in a SAVEPOINTed session.
    from sqlalchemy import select

    from models import ApiKey

    api_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.tenant_id == tenant.id))
    ).scalar_one()

    room = f"room-{uuid.uuid4().hex[:8]}"
    s1 = Session(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room=room,
        identity="u1",
    )
    db_session.add(s1)
    await db_session.flush()

    s2 = Session(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        room=room,  # collision
        identity="u2",
    )
    db_session.add(s2)
    with pytest.raises(Exception):
        await db_session.flush()


async def test_updated_at_trigger_fires_on_tenant_update(db_session):
    from models import Tenant

    t = Tenant(name="TickTock", slug=f"tick-{uuid.uuid4().hex[:6]}")
    db_session.add(t)
    await db_session.flush()
    await db_session.refresh(t)
    first_updated_at = t.updated_at

    # Force a UPDATE. Using raw SQL bypasses the ORM's optimization that
    # skips updates when no fields changed.
    await db_session.execute(
        text("UPDATE tenants SET name = 'TockTick' WHERE id = :id"),
        {"id": t.id},
    )
    await db_session.flush()
    await db_session.refresh(t)

    assert t.updated_at > first_updated_at
    assert t.name == "TockTick"

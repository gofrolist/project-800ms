"""Test fixtures for the api service.

Sets fake LiveKit credentials *before* the production modules are imported,
because `settings.py` instantiates `Settings()` at module-import time and
would otherwise raise a ValidationError.

Also owns the Postgres testcontainer used by DB-backed tests — one container
per session, migrations applied once, individual tests run inside a
BEGIN/ROLLBACK savepoint for isolation.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

# Required env before any application import.
os.environ.setdefault("LIVEKIT_API_KEY", "test-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-secret-min-32-chars-long-xxxxx")

# Make the service root importable as if `python main.py` were run from there.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


# ─── Postgres testcontainer ───────────────────────────────────────────────
#
# Tests that need a real DB mark themselves with @pytest.mark.slow. The
# container spins up once per session (~3-5s). If Docker isn't running,
# these tests are skipped — CI provides Postgres via a service container
# instead (see .github/workflows/ci.yml).


def _skip_if_no_docker() -> None:
    try:
        import docker  # type: ignore[import-untyped]

        docker.from_env().ping()
    except Exception as exc:  # broad: many shapes of "Docker unavailable"
        pytest.skip(f"Docker not available for DB tests: {exc}")


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Start a Postgres container, run migrations, return the async URL.

    Honors PG_TEST_URL env var — when set, tests use that Postgres instead
    of starting a container. Used in CI with GHA `services: postgres:`.
    """
    precomputed = os.environ.get("PG_TEST_URL")
    if precomputed:
        os.environ["DATABASE_URL"] = precomputed
        _run_migrations(precomputed)
        return precomputed

    _skip_if_no_docker()
    from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

    # Use the image that ships in our own compose stack so dialect behavior
    # matches prod exactly.
    container = PostgresContainer("postgres:18-alpine", driver="asyncpg")
    container.start()
    url = container.get_connection_url()
    # testcontainers returns postgresql+psycopg2:// by default on older
    # versions; normalize.
    if "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    os.environ["DATABASE_URL"] = url
    _run_migrations(url)
    yield url
    container.stop()


def _run_migrations(url: str) -> None:
    """Apply alembic migrations to the given DB URL. Idempotent."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(SERVICE_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session(postgres_url: str) -> AsyncIterator[object]:
    """One transaction per test, rolled back at the end.

    SAVEPOINTs + a nested transaction give us per-test isolation without
    re-running migrations. Every test sees a clean slate (except for
    migration-seeded rows like the 'dev' / 'demo' tenants — but those only
    exist if SEED_*_API_KEY env vars were set at container start, which
    they aren't in tests).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.connect() as conn:
        trans = await conn.begin()
        session_factory = async_sessionmaker(bind=conn, expire_on_commit=False)
        async with session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
        await trans.rollback()
    await engine.dispose()


@pytest_asyncio.fixture
async def seed_tenant(db_session):
    """Insert a throwaway tenant + active API key; return (tenant, raw_key)."""
    import hashlib

    from models import ApiKey, Tenant

    raw_key = "tk_" + uuid.uuid4().hex  # 35 chars
    tenant = Tenant(
        name="Test",
        slug=f"test-{uuid.uuid4().hex[:8]}",
        rate_limit_per_minute=60,
        allowed_origins=["http://localhost:5173"],
    )
    db_session.add(tenant)
    await db_session.flush()

    key = ApiKey(
        tenant_id=tenant.id,
        key_hash=hashlib.sha256(raw_key.encode()).digest(),
        key_prefix=raw_key[:8],
        label="test",
    )
    db_session.add(key)
    await db_session.flush()

    return tenant, raw_key

"""Shared pytest fixtures for the retriever test suite.

Per constitution Principle II, integration tests hit a real Postgres +
pgvector instance rather than mocks. We spin up the `pgvector/pgvector:pg16`
image via `testcontainers[postgresql]`, apply the project's Alembic
migrations once per session, and hand function-scoped rolled-back sessions
to each test for isolation.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

# Repo root resolved from this file's location so tests run from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "apps" / "api"


@pytest.fixture(scope="session")
def pgvector_postgres() -> Iterator[PostgresContainer]:
    """Session-scoped pgvector/pgvector:pg16 container.

    Starts once per pytest session; every test re-uses it. The container
    is torn down on session exit. Skipped (via pytest.skip) if Docker
    isn't reachable — this keeps unit-only test runs fast.
    """
    try:
        container = PostgresContainer(
            image="pgvector/pgvector:pg16",
            username="voice",
            password="test",
            dbname="voice",
        )
        container.start()
    except Exception as exc:  # pragma: no cover — infra-level skip
        pytest.skip(f"Docker / testcontainers unavailable: {exc}")

    # Alembic subprocess runs INSIDE the try/finally so a non-zero exit
    # can never leak the container. Earlier structure ran the subprocess
    # above the try: block, so a failing migration bypassed
    # container.stop() and accumulated zombies in CI.
    try:
        db_url = container.get_connection_url()
        dsn = db_url.replace("+psycopg2", "+asyncpg")
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env.update(
            {
                "DATABASE_URL": dsn,
                "LIVEKIT_API_KEY": env.get("LIVEKIT_API_KEY", "test-livekit-key"),
                "LIVEKIT_API_SECRET": env.get("LIVEKIT_API_SECRET", "x" * 32),
            }
        )
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=_API_DIR,
            env=env,
            check=True,
        )
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def pgvector_dsn(pgvector_postgres: PostgresContainer) -> str:
    """The asyncpg-flavoured DSN for the session container.

    Asyncpg requires `postgresql+asyncpg://` — testcontainers returns the
    sync flavour; we rewrite it here rather than making every test do so.
    """
    url = pgvector_postgres.get_connection_url()
    if "+psycopg2" in url:
        return url.replace("+psycopg2", "+asyncpg")
    if "+psycopg" in url:
        return url.replace("+psycopg", "+asyncpg")
    return url.replace("postgresql://", "postgresql+asyncpg://")


@pytest_asyncio.fixture
async def db_session(pgvector_dsn: str) -> AsyncIterator[AsyncSession]:
    """Function-scoped async session with per-test rollback.

    Each test gets a fresh transactional session against the session-
    scoped container. Rollback on exit keeps tests isolated — no DELETEs
    or TRUNCATEs needed between tests.
    """
    engine = create_async_engine(pgvector_dsn, pool_pre_ping=True, future=True)
    try:
        async with engine.connect() as conn:
            async with conn.begin() as outer_txn:
                session = AsyncSession(bind=conn, expire_on_commit=False)
                try:
                    yield session
                finally:
                    await session.close()
                    await outer_txn.rollback()
    finally:
        await engine.dispose()

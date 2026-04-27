"""Shared pytest fixtures for the retriever test suite.

Per constitution Principle II, integration tests hit a real Postgres +
pgvector instance rather than mocks. We spin up the `pgvector/pgvector:pg18`
image via `testcontainers[postgresql]`, apply the project's Alembic
migrations once per session, and hand function-scoped rolled-back sessions
to each test for isolation.

The session-scoped `retriever_app` fixture builds a fresh FastAPI app
pointed at the testcontainer with one tenant + session + KB seeded.
US1's test_retrieve_endpoint, US2's test_refusal_path / test_timing_parity
all reuse it to avoid spinning up multiple containers.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

# Repo root resolved from this file's location so tests run from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "apps" / "api"


@pytest.fixture(scope="session", autouse=True)
def _retriever_env_baseline() -> Iterator[None]:
    """Session-level baseline env so any test that imports the
    embedder + Settings doesn't blow up on missing config.

    Uses ``os.environ.setdefault`` so a real value passed in by the
    operator (e.g. via direnv or an interactive shell) wins. The
    function-scoped ``retriever_app`` fixture later overrides DB_URL
    with the testcontainer DSN via monkeypatch — this baseline only
    keeps the FIRST ``preload()`` call from raising while the
    container is still booting.
    """
    os.environ.setdefault("DB_URL", "postgresql+asyncpg://placeholder/placeholder")
    os.environ.setdefault("LLM_BASE_URL", "http://llm-test.local")
    os.environ.setdefault("LLM_API_KEY", "test-key")
    os.environ.setdefault("REWRITER_MODEL", "test-rewriter-model")
    os.environ.setdefault("EMBEDDER_DEVICE", "cpu")
    # Issue #40 / #47: the auth dep returns 503 when this is empty so
    # /retrieve refuses traffic — set a non-secret default for tests.
    os.environ.setdefault("RETRIEVER_INTERNAL_TOKEN", "test-internal-token")
    yield


@pytest.fixture(scope="session")
def pgvector_postgres() -> Iterator[PostgresContainer]:
    """Session-scoped pgvector/pgvector:pg18 container.

    Starts once per pytest session; every test re-uses it. The container
    is torn down on session exit. Skipped (via pytest.skip) if Docker
    isn't reachable — this keeps unit-only test runs fast.
    """
    try:
        container = PostgresContainer(
            image="pgvector/pgvector:pg18",
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

    Issue #44: ``join_transaction_mode='create_savepoint'`` makes the
    AsyncSession's own ``commit()`` translate to a SAVEPOINT release
    rather than committing the outer transaction. Without it, a test
    that explicitly commits would defeat the rollback isolation —
    fine today (no test commits) but a lurking trap for any Phase 3
    code that grows real upsert flows.
    """
    engine = create_async_engine(pgvector_dsn, pool_pre_ping=True, future=True)
    try:
        async with engine.connect() as conn:
            async with conn.begin() as outer_txn:
                session = AsyncSession(
                    bind=conn,
                    expire_on_commit=False,
                    join_transaction_mode="create_savepoint",
                )
                try:
                    yield session
                finally:
                    await session.close()
                    await outer_txn.rollback()
    finally:
        await engine.dispose()


# ─── Retriever app fixture (shared by integration + refusal + parity tests) ─────
#
# Seeds one tenant + session + 3 KB chunks against the session-scoped
# container, then builds a fresh FastAPI app pointed at the same DSN.
# All tests sharing this fixture rely on the LLM rewriter being mocked
# via respx; the LLM_BASE_URL env points at a non-routable host so no
# accidental real network call can succeed.

_FIXTURE_LLM_BASE = "http://llm-test.local"


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10g}" for x in v) + "]"


@pytest_asyncio.fixture
async def retriever_app(pgvector_postgres, monkeypatch) -> dict[str, Any]:
    """Build a fresh FastAPI app pointed at the session-scoped pgvector
    container, and seed one tenant + session + KB entry + 3 chunks.

    Seed is COMMITTED on a dedicated engine so the app's own engine
    (opened by ``db.get_engine()``) sees the data. The shared container
    tears down at session end — each test uses unique UUIDs to avoid
    collisions with other tests touching the same tables.

    Returns a dict with keys:
        ``app``        — the FastAPI app instance.
        ``tenant_id``  — the seeded tenant UUID.
        ``session_id`` — the seeded session UUID.
        ``kb_entry_id`` — the seeded kb_entry UUID.
        ``llm_base``   — base URL the rewriter LLM is configured against
                         (so respx can install matchers).
    """
    dsn = pgvector_postgres.get_connection_url().replace("+psycopg2", "+asyncpg")

    # Env must be set BEFORE preload() — Settings() is a Pydantic
    # BaseSettings that fails-fast on missing required fields, and
    # preload() instantiates it transitively via get_settings().
    monkeypatch.setenv("DB_URL", dsn)
    monkeypatch.setenv("LLM_BASE_URL", _FIXTURE_LLM_BASE)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("REWRITER_MODEL", "test-rewriter-model")
    monkeypatch.setenv("EMBEDDER_DEVICE", "cpu")
    monkeypatch.setenv("RETRIEVER_INTERNAL_TOKEN", "test-internal-token")

    import config

    config.get_settings.cache_clear()

    from embedder import encode, preload

    preload()

    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    api_key_id = uuid.uuid4()
    kb_entry_id = uuid.uuid4()

    chunk_specs: list[dict[str, Any]] = [
        {
            "section": "Получение прав",
            "content": (
                "Чтобы получить водительские права, посетите автошколу "
                "и сдайте экзамены. Стоимость обучения 5 000 долларов."
            ),
        },
        {
            "section": "Цены на транспорт",
            "content": (
                "Грузовик Mule стоит 15 000 долларов. "
                "Легковой автомобиль Premier стоит 12 500 долларов."
            ),
        },
        {
            "section": "Покупка оружия",
            "content": (
                "Оружие можно купить в магазине после получения лицензии. "
                "Лицензия стоит 10 000 долларов."
            ),
        },
    ]
    for spec in chunk_specs:
        spec["embedding"] = _vector_literal(await encode(spec["content"]))

    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, slug) VALUES (:i, :n, :s)"),
            {"i": tenant_id, "n": "Test Tenant", "s": f"t-{tenant_id.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, tenant_id, key_hash, key_prefix) "
                "VALUES (:i, :t, decode(:h, 'hex'), :p)"
            ),
            {
                "i": api_key_id,
                "t": tenant_id,
                "h": secrets.token_hex(32),
                "p": f"pfx{tenant_id.hex[:5]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO sessions (id, tenant_id, api_key_id, room, identity) "
                "VALUES (:si, :t, :ai, :room, 'test-user')"
            ),
            {
                "si": session_id,
                "t": tenant_id,
                "ai": api_key_id,
                "room": f"room-{session_id.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO kb_entries "
                "(id, tenant_id, kb_entry_key, title, content_sha256) "
                "VALUES (:i, :t, 'main', 'Main', :sha)"
            ),
            {"i": kb_entry_id, "t": tenant_id, "sha": "shadeadbeef"},
        )
        for idx, spec in enumerate(chunk_specs):
            await conn.execute(
                text(
                    "INSERT INTO kb_chunks "
                    "(tenant_id, kb_entry_id, section, title, content, "
                    " content_sha256, embedding) "
                    "VALUES (:t, :e, :sec, :title, :content, :sha, "
                    "        CAST(:emb AS vector))"
                ),
                {
                    "t": tenant_id,
                    "e": kb_entry_id,
                    "sec": spec["section"],
                    "title": f"Главный раздел — {spec['section']}",
                    "content": spec["content"],
                    "sha": f"sha-chunk-{idx}",
                    "emb": spec["embedding"],
                },
            )
    await engine.dispose()

    # Env was already set above. Clear cached settings + DB factories
    # so the app picks up the now-correct DSN and the (still-correct)
    # rewriter / embedder env.
    import db
    import hybrid_search as hs

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()
    # T038: latency stats are kept in module-level state — wipe between
    # tests so a prior test's recorded p50 doesn't bleed into the next.
    if hasattr(hs, "_reset_latency_stats"):
        hs._reset_latency_stats()

    import sys

    sys.modules.pop("main", None)
    sys.modules.pop("retrieve", None)
    from main import create_app

    app = create_app()

    yield {
        "app": app,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "kb_entry_id": kb_entry_id,
        "llm_base": _FIXTURE_LLM_BASE,
        "internal_token": "test-internal-token",
        "auth_headers": {"X-Internal-Token": "test-internal-token"},
    }

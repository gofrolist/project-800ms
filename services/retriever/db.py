"""Async Postgres connection pool for the retriever.

Thin wrapper around SQLAlchemy 2.0 async + asyncpg. Exports a lazily-
constructed `AsyncEngine` and a `get_session()` async context manager.

Tenant-scoping (constitution Principle IV) is enforced at the call site:
every retriever query that reads tenant data carries `WHERE tenant_id = $1`
explicitly. An earlier draft shipped `fetch_*_tenant_scoped` helpers that
claimed to enforce this via a positional-only `tenant_id` arg, but (a) the
helpers had no callers, (b) the SQL was still free-form so the WHERE
clause was still the caller's responsibility, and (c) a `tenant_id=` kwarg
in `**params` could shadow the positional. Deleted rather than kept as
misleading scaffolding. Phase 3 needs a real abstraction (repository class
or SQL AST requiring a tenant predicate) if this guarantee ever becomes a
type-level one.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide SQLAlchemy async engine.

    Lazy singleton — first call pays the pool creation cost. Subsequent
    calls are free. Tests patching `DB_URL` should clear both
    `get_engine.cache_clear()` and `_session_factory.cache_clear()`.
    """
    settings = get_settings()
    return create_async_engine(
        settings.db_url,
        # Sensible pool size for a single retriever replica doing
        # ~100 QPS in the worst case. Tune if we observe pool starvation.
        pool_size=20,
        max_overflow=10,
        # pool_timeout=2 aligns with the retrieval SLO budget (≤500 ms).
        # Default 30s would manifest as a silent hang, not a fast 503.
        pool_timeout=2,
        # Recycle idle conns every 30 min to avoid middlebox (PgBouncer,
        # managed-DB connection) timeouts.
        pool_recycle=1800,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding an `AsyncSession`.

    Commits on success, rolls back on exception. Every call site that
    queries tenant data MUST include `WHERE tenant_id = :tenant_id` in
    its SQL AND must call ``set_tenant_scope(session, tenant_id)``
    before any SELECT/INSERT against the RLS-protected tables
    (``kb_entries``, ``kb_chunks``, ``retrieval_traces``). The
    application-layer predicate is the fast path; the RLS policy
    (migration 0006) is the data-layer backstop that blocks rows the
    predicate forgot to filter.
    """
    factory = _session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def set_tenant_scope(session: AsyncSession, tenant_id: object) -> None:
    """Set the per-transaction ``app.current_tenant_id`` GUC that the
    RLS policy on ``kb_entries`` / ``kb_chunks`` / ``retrieval_traces``
    consults (issue #41 / migration 0006).

    Uses ``set_config(name, value, is_local=true)`` so the GUC
    auto-resets when the current transaction commits — a connection
    returned to the pool cannot leak its scope to the next request.
    Forgetting this call makes every RLS-protected query return zero
    rows (fail-closed).

    ``tenant_id`` accepts ``str`` or ``uuid.UUID``; both stringify to
    the canonical hex form Postgres accepts in the policy's COALESCE.
    """
    from sqlalchemy import text

    await session.execute(
        text("SELECT set_config('app.current_tenant_id', :tid, true)"),
        {"tid": str(tenant_id)},
    )

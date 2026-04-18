"""Async SQLAlchemy engine + session factory + FastAPI dependency.

One engine per process, created at import time. `get_db()` yields a session
per request and rolls back on unhandled exceptions — the standard FastAPI
pattern for SQLAlchemy 2.0 async.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from settings import settings

# echo=False in prod (logs would leak SQL params on error). Set
# SQLALCHEMY_ECHO=true via env + re-read Settings to enable for debugging.
_engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

async_session_factory = async_sessionmaker(
    _engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — one session per request, rollback on error."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

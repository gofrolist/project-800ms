"""Retriever service FastAPI app entrypoint.

This module wires the FastAPI app + routes. The heavy lifting (hybrid
search, rewriter, embedder, trace writer) lives in sibling modules.

Spec: specs/002-helper-guide-npc/contracts/retrieve.openapi.yaml
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import text

import embedder
from config import get_settings
from db import get_engine
from errors import register_exception_handlers
from health import router as health_router
from logging_setup import configure_logging

# Guard for `configure_logging` so repeat calls from `create_app()` in
# tests don't stack loguru sinks. Flipped on first successful configure.
_logging_configured = False


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Boot-time side effects: preload the BGE-M3 embedder singleton and
    warm the DB pool so `/ready` can flip healthy before first traffic.

    Constitution Principle V: heavy models MUST be preloaded at boot, not
    lazily on first turn. Without this, compose's healthcheck never goes
    green and the agent's eventual dependency on retriever blocks forever.
    """
    try:
        embedder.preload()
    except Exception:
        logger.exception("lifespan.embedder_preload_failed")
        raise

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        logger.warning(
            "lifespan.db_warmup_failed message={message}",
            message="DB not reachable at boot; /ready will report degraded",
        )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app. Kept as a factory so tests can instantiate
    isolated app objects without reusing module-level state."""
    global _logging_configured
    settings = get_settings()
    if not _logging_configured:
        configure_logging(settings.log_level)
        _logging_configured = True

    # /docs and /openapi.json are internal-only but still a public-surface
    # disclosure risk if compose mappings ever change. Gate behind the
    # same DEBUG signal as verbose logging — operators opt in explicitly.
    docs_enabled = settings.log_level == "DEBUG"

    app = FastAPI(
        title="project-800ms retriever",
        version="0.1.0",
        docs_url="/docs" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        redoc_url=None,
        lifespan=_lifespan,
    )

    app.include_router(health_router)
    # /retrieve is registered in Phase 3 (US1, task T027).

    register_exception_handlers(app)

    return app


app = create_app()

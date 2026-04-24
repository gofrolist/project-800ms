"""Liveness and readiness probes.

`/healthz` — process is up. Returns 200 unconditionally. Used by orchestrators
   that only need to know the container didn't crash.

`/ready`   — the service is actually able to serve `/retrieve`. Verifies:
   (a) embedder singleton is loaded,
   (b) DB pool can execute `SELECT 1`,
   (c) rewriter LLM answered a hello-ping within the last 30 s (cached).
   Used by the docker-compose healthcheck (start_period: 600s — matches
   HF_HUB_DOWNLOAD_TIMEOUT so a slow cold download can't trigger a
   kill/re-download boot loop).

Exception logging on this endpoint is deliberately narrow: the message is
`type(exc).__name__` only. asyncpg / SQLAlchemy error messages can embed
the full DSN (including the DB password), which would be shipped to any
SIEM that ingests container logs.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, Response
from loguru import logger
from sqlalchemy import text

from config import get_settings
from db import get_engine
from embedder import is_loaded as embedder_is_loaded

router = APIRouter(tags=["health"])

_REWRITER_CACHE_TTL_SECONDS = 30.0
# The probe itself has a budget well inside the compose healthcheck
# timeout (3 s). Keeping it at 2.5 s leaves 500 ms headroom.
_REWRITER_PROBE_TIMEOUT = httpx.Timeout(2.5, connect=1.0)


@dataclass
class _RewriterCache:
    ok: bool = False
    ts: float = float("-inf")  # 'never probed' sentinel distinct from 0.0


_rewriter_cache = _RewriterCache()
# Serialises the in-flight probe so N concurrent /ready calls produce
# one LLM POST, not N. Defeats the cold-start thundering-herd.
_rewriter_lock = asyncio.Lock()


def reset_rewriter_cache() -> None:
    """Test seam — clear the cached rewriter-probe result."""
    _rewriter_cache.ok = False
    _rewriter_cache.ts = float("-inf")


@router.get("/healthz", include_in_schema=True)
async def healthz() -> dict[str, str]:
    """Liveness. No dependency checks — returns as long as the process
    can serve HTTP."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(response: Response) -> dict[str, object]:
    """Readiness. Verifies embedder + DB + rewriter (cached)."""
    checks: dict[str, bool] = {}

    checks["embedder"] = embedder_is_loaded()

    # DB pool can serve a trivial query. Narrow the exception logging
    # message so the DSN + password never reach logs.
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception as exc:  # noqa: BLE001 — logged + recovered
        logger.warning("ready.db_check_failed kind={kind}", kind=type(exc).__name__)
        checks["db"] = False

    checks["rewriter"] = await _check_rewriter_cached()

    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return {"status": "ok" if ok else "degraded", "checks": checks}


async def _check_rewriter_cached() -> bool:
    """Probe the rewriter LLM with a trivial call, cached for 30 s.

    Serialised by `_rewriter_lock` so only one probe runs at a time — the
    classic cold-start thundering-herd where N concurrent /ready calls
    would otherwise fan out N real LLM POSTs.
    """
    now = time.monotonic()
    if now - _rewriter_cache.ts < _REWRITER_CACHE_TTL_SECONDS:
        return _rewriter_cache.ok

    async with _rewriter_lock:
        # Another probe may have refreshed the cache while we were
        # waiting for the lock.
        now = time.monotonic()
        if now - _rewriter_cache.ts < _REWRITER_CACHE_TTL_SECONDS:
            return _rewriter_cache.ok

        settings = get_settings()
        url = f"{str(settings.llm_base_url).rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        payload = {
            "model": settings.rewriter_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        try:
            async with httpx.AsyncClient(timeout=_REWRITER_PROBE_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, json=payload)
            # Treat auth / not-found / rate-limit as NOT ready — a revoked
            # key should page, not silently green-light the probe.
            status = resp.status_code
            ok = 200 <= status < 500 and status not in (401, 403, 404, 429)
            if not ok:
                logger.warning("ready.rewriter_non_ok status={status}", status=status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ready.rewriter_check_failed kind={kind}", kind=type(exc).__name__)
            ok = False

        _rewriter_cache.ok = ok
        _rewriter_cache.ts = now
        return ok

"""Liveness and readiness probes.

`/healthz` — process is up. Returns 200 unconditionally. Used by orchestrators
   that only need to know the container didn't crash.

`/ready`   — the service is actually able to serve `/retrieve`. Verifies:
   (a) embedder singleton is loaded,
   (b) DB pool can execute `SELECT 1`,
   (c) the rewriter LLM endpoint is reachable (cached probe — see
       `_check_rewriter_cached`).
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

# Two-tier cache for the rewriter probe. Successful probes are trusted for 5
# minutes; failed probes for 1 minute. The asymmetry is deliberate:
#
# * 300 s on success keeps the docker healthcheck (10 s interval) below the
#   provider's per-day request budget on free tiers — Groq's 1000 RPD on
#   `/chat/completions` is the bound that bit us when this was 30 s and
#   billable; switching to the unbilled `/models` GET (see
#   `_check_rewriter_cached`) plus the 5-min cache puts us at 288/day.
# * 60 s on failure prevents a single transient hiccup (network blip, brief
#   per-minute rate-limit window, upstream 5xx) from poisoning the cache for
#   the full 5-minute window. A revoked key still surfaces within one minute.
_REWRITER_CACHE_TTL_OK = 300.0
_REWRITER_CACHE_TTL_FAIL = 60.0
# HTTP statuses that indicate the rewriter is NOT ready and the cached
# negative result should page operators on the next /ready call. Auth
# failures (401/403), wrong base URL (404), and quota exhaustion (429) are
# operator-visible problems, not transient network issues.
_REWRITER_NOT_READY_STATUSES = frozenset({401, 403, 404, 429})
# The probe itself has a budget well inside the compose healthcheck
# timeout (3 s). Keeping it at 2.5 s leaves 500 ms headroom.
_REWRITER_PROBE_TIMEOUT = httpx.Timeout(2.5, connect=1.0)


@dataclass
class _RewriterCache:
    ok: bool = False
    ts: float = float("-inf")  # 'never probed' sentinel distinct from 0.0


_rewriter_cache = _RewriterCache()
# Serialises the in-flight probe so N concurrent /ready calls produce
# one LLM request, not N. Defeats the cold-start thundering-herd.
_rewriter_lock = asyncio.Lock()


def reset_rewriter_cache() -> None:
    """Test seam — clear the cached rewriter-probe result and reset the lock.

    The lock reset is defensive: a test that cancels a coroutine mid-probe
    would otherwise leave `_rewriter_lock` permanently held in module state,
    deadlocking every subsequent test that touches `_check_rewriter_cached`.
    """
    global _rewriter_lock
    _rewriter_cache.ok = False
    _rewriter_cache.ts = float("-inf")
    _rewriter_lock = asyncio.Lock()


def _is_cache_fresh(now: float) -> bool:
    """Two-tier cache freshness. Reads `_rewriter_cache.ok` (no lock; safe
    in single-threaded asyncio) and picks the matching TTL."""
    ttl = _REWRITER_CACHE_TTL_OK if _rewriter_cache.ok else _REWRITER_CACHE_TTL_FAIL
    return now - _rewriter_cache.ts < ttl


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
    """Probe the rewriter LLM endpoint with a metadata GET, cached.

    Hits ``GET {llm_base_url}/models`` — part of the OpenAI-compatible
    surface that Groq, vLLM, and OpenAI all implement, and (unlike
    ``POST /chat/completions``) unbilled on every provider we target.
    Cache TTLs are asymmetric (see module-level constants).

    Serialised by `_rewriter_lock` so concurrent /ready calls produce
    one in-flight request, not N.
    """
    now = time.monotonic()
    if _is_cache_fresh(now):
        return _rewriter_cache.ok

    async with _rewriter_lock:
        # Another probe may have refreshed the cache while we were
        # waiting for the lock.
        now = time.monotonic()
        if _is_cache_fresh(now):
            return _rewriter_cache.ok

        settings = get_settings()
        url = f"{str(settings.llm_base_url).rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_REWRITER_PROBE_TIMEOUT) as client:
                resp = await client.get(url, headers=headers)
            status = resp.status_code
            ok = 200 <= status < 500 and status not in _REWRITER_NOT_READY_STATUSES
            if not ok:
                logger.warning("ready.rewriter_non_ok status={status}", status=status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ready.rewriter_check_failed kind={kind}", kind=type(exc).__name__)
            ok = False

        _rewriter_cache.ok = ok
        _rewriter_cache.ts = now
        return ok

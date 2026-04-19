"""Rate limiting — IP-based for unauthenticated paths, per-tenant for /v1/*.

Two separate mechanisms live here:

1. slowapi's `limiter` — still drives /health. IP-based, fixed rate.
   slowapi's `@limit` decorator only accepts static rate strings or
   no-arg callables, which is why it can't represent "use whatever rate
   the tenant has in their DB row".

2. `enforce_tenant_rate_limit` dep — reads `rate_limit_per_minute` off
   the authenticated tenant's identity and consumes a token from a
   per-tenant in-memory bucket. Chained after `enforce_tenant_origin`
   so origin enforcement still runs first.

The buckets are process-local. Scaling beyond one api replica will need
a shared backend (Redis / Memcached with an atomic compare-and-swap),
but for the current single-box deploy this is both fast and correct.

Token-bucket semantics: each tenant starts at their configured rate,
refills linearly at `rate / 60` tokens/second (cap: full rate). A
request consumes one token; if none are available we return 429.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

from cachetools import TTLCache
from fastapi import Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from auth import TenantIdentity, enforce_tenant_origin
from errors import APIError


def _real_ip(request: Request) -> str:
    """Extract the real client IP when behind Caddy (matches legacy behavior)."""
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    if xff and client_host.startswith("172."):
        return xff.split(",")[0].strip()
    return get_remote_address(request)


def tenant_or_ip(request: Request) -> str:
    """slowapi limiter key — prefer tenant id when the auth dep already
    ran, fall back to IP otherwise. Only /health uses this today."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    return f"ip:{_real_ip(request)}"


limiter = Limiter(key_func=tenant_or_ip, default_limits=[])


# ─── Per-tenant token bucket ──────────────────────────────────────────────


@dataclass
class _TokenBucket:
    """Classic token bucket. One per tenant.

    Keeps state behind a lock because the bucket is shared by all
    concurrent requests for this tenant in this process. asyncio is
    single-threaded but we use threads for sync dependencies, so a
    Lock is the safe primitive here.
    """

    capacity: float
    tokens: float
    refill_per_sec: float
    last_refill: float
    lock: Lock

    def consume(self, cost: float = 1.0) -> bool:
        """Try to consume `cost` tokens. Return True if granted."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            if elapsed > 0:
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.refill_per_sec,
                )
                self.last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    @classmethod
    def for_rate(cls, rate_per_minute: int) -> _TokenBucket:
        return cls(
            capacity=float(rate_per_minute),
            tokens=float(rate_per_minute),
            refill_per_sec=rate_per_minute / 60.0,
            last_refill=time.monotonic(),
            lock=Lock(),
        )


# Buckets live in a TTL cache so tenants we haven't seen in a while get
# garbage-collected. 10 min TTL = idle tenants lose their bucket state
# (they'll just start full again on the next request, which is fine).
_buckets: TTLCache[str, _TokenBucket] = TTLCache(maxsize=4096, ttl=600)
_buckets_lock = Lock()


def _get_bucket(tenant_id: str, rate_per_minute: int) -> _TokenBucket:
    """Fetch or create a bucket, resetting capacity if the tenant's rate
    changed since last request."""
    with _buckets_lock:
        bucket = _buckets.get(tenant_id)
        if bucket is None or bucket.capacity != rate_per_minute:
            bucket = _TokenBucket.for_rate(rate_per_minute)
            _buckets[tenant_id] = bucket
        return bucket


def _reset_buckets_for_tests() -> None:
    """Tests only — drop all cached buckets so rate-limit tests start fresh."""
    with _buckets_lock:
        _buckets.clear()


async def enforce_tenant_rate_limit(
    identity: TenantIdentity = Depends(enforce_tenant_origin),
) -> TenantIdentity:
    """Per-tenant token-bucket rate limit.

    The `rate_limit_per_minute` comes straight off `TenantIdentity`, so
    admin edits via `PATCH /v1/admin/tenants/{slug}` take effect as soon
    as the auth cache TTL flushes (default 60s).

    Raises:
        APIError(429) — bucket empty for this tenant.
    """
    bucket = _get_bucket(identity.tenant_id, identity.rate_limit_per_minute)
    if not bucket.consume():
        raise APIError(
            429,
            "rate_limited",
            f"Rate limit exceeded: {identity.rate_limit_per_minute}/minute",
        )
    return identity

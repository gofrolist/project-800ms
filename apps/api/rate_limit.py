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

import ipaddress
import time
from dataclasses import dataclass
from threading import Lock

from cachetools import TTLCache
from fastapi import Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from auth import TenantIdentity, enforce_tenant_origin
from errors import APIError
from settings import settings


def _is_trusted_proxy(client_host: str) -> bool:
    """True when the direct TCP peer sits inside a configured proxy CIDR.

    Only when this is true do we trust X-Forwarded-For. This replaces a
    previous `startswith("172.")` heuristic which matched the entire
    172.0.0.0/8 block (not just RFC1918 172.16.0.0/12) and, more
    importantly, was a string prefix check rather than a real network
    membership test. Returns False on bad input so no exception reaches
    a request handler — rate limiting continues to work keyed on the
    peer address instead.
    """
    if not client_host:
        return False
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    for cidr in settings.trusted_proxy_cidrs:
        try:
            if client_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def _real_ip(request: Request) -> str:
    """Extract the real client IP when behind a trusted reverse proxy.

    X-Forwarded-For is honoured only when the TCP peer's address falls
    inside settings.trusted_proxy_cidrs. In every other case we key on
    the peer address directly — preventing a Docker-network attacker
    from rotating XFF to unlock per-request IP-keyed buckets.
    """
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    if xff and _is_trusted_proxy(client_host):
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


# Two separate caches so an unauth-IP flood can't LRU-evict authenticated
# tenant rate-limit state. The tenant cache is sized for the expected max
# live tenants on a single-box deploy; the IP cache is larger because
# webhook / admin traffic legitimately spans many distinct IPs and we
# don't want distinct-IP churn to evict still-active buckets. Keeping
# both under _buckets_lock for test teardown simplicity.
_tenant_buckets: TTLCache[str, _TokenBucket] = TTLCache(maxsize=4096, ttl=600)
_ip_buckets: TTLCache[str, _TokenBucket] = TTLCache(maxsize=32768, ttl=600)
_buckets_lock = Lock()


def _get_bucket(cache: TTLCache[str, _TokenBucket], key: str, rate_per_minute: int) -> _TokenBucket:
    """Fetch or create a bucket in the given cache, resetting capacity if
    the caller-configured rate changed since last request."""
    with _buckets_lock:
        bucket = cache.get(key)
        if bucket is None or bucket.capacity != rate_per_minute:
            bucket = _TokenBucket.for_rate(rate_per_minute)
            cache[key] = bucket
        return bucket


def _reset_buckets_for_tests() -> None:
    """Tests only — drop all cached buckets so rate-limit tests start fresh."""
    with _buckets_lock:
        _tenant_buckets.clear()
        _ip_buckets.clear()


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
    bucket = _get_bucket(_tenant_buckets, identity.tenant_id, identity.rate_limit_per_minute)
    if not bucket.consume():
        raise APIError(
            429,
            "rate_limited",
            f"Rate limit exceeded: {identity.rate_limit_per_minute}/minute",
        )
    return identity


# ─── IP-based limits (unauth paths) ───────────────────────────────────────
#
# Buckets keyed on IP for routes that don't carry a tenant identity.
# Defense-in-depth: even a constant-time key check can be brute-forced
# online if nothing caps the request rate. Limits are tunable via
# settings.admin_ip_rate_per_minute / settings.webhook_ip_rate_per_minute
# so ops can widen them for bulk automation (Terraform, CI) without
# touching code.


async def enforce_admin_ip_rate_limit(request: Request) -> None:
    """IP-based rate limit for /v1/admin/*.

    Runs *before* the admin-key check, so an attacker spraying bad keys
    burns their rate budget without ever reaching `secrets.compare_digest`.

    Short-circuits when `settings.admin_api_key` is empty — the admin
    surface is already disabled in that case (every route returns 503
    via `_require_admin`), so consuming an IP-bucket token would only
    punish legitimate operators on a shared egress IP and let attackers
    thrash the cache by probing a disabled endpoint.
    """
    if not settings.admin_api_key:
        return
    ip = _real_ip(request)
    bucket = _get_bucket(_ip_buckets, f"admin-ip:{ip}", settings.admin_ip_rate_per_minute)
    if not bucket.consume():
        raise APIError(429, "rate_limited", "Rate limit exceeded")


async def enforce_webhook_ip_rate_limit(request: Request) -> None:
    """IP-based rate limit for /v1/livekit-webhook.

    JWT verification is cheap but not free; an attacker flooding invalid
    signatures could burn CPU + log noise. LiveKit sends from a small
    set of source IPs, and a single room generates well under 1000
    events/min even under churn.
    """
    ip = _real_ip(request)
    bucket = _get_bucket(_ip_buckets, f"webhook-ip:{ip}", settings.webhook_ip_rate_per_minute)
    if not bucket.consume():
        raise APIError(429, "rate_limited", "Rate limit exceeded")

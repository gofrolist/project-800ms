"""slowapi limiter configured for tenant-aware keys.

Rate-limiting key order:
    1. request.state.tenant_id — populated by auth when the dep runs
    2. X-Forwarded-For (first hop) when behind Caddy
    3. Peer IP

Each tenant has its own `rate_limit_per_minute` column; handlers apply
the per-tenant value dynamically via `limiter.limit(tenant_rate_limit)`.
Routes that don't take auth (health, docs) continue to use IP-based
limits.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _real_ip(request: Request) -> str:
    """Extract the real client IP when behind Caddy (matches legacy behavior)."""
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    if xff and client_host.startswith("172."):
        return xff.split(",")[0].strip()
    return get_remote_address(request)


def tenant_or_ip(request: Request) -> str:
    """Limiter key — prefer tenant id when authenticated, fall back to IP."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    return f"ip:{_real_ip(request)}"


limiter = Limiter(key_func=tenant_or_ip, default_limits=[])


def tenant_rate_limit(request: Request) -> str:
    """Per-tenant minute budget, resolved from request.state.

    Used as `@limiter.limit(tenant_rate_limit)` on /v1/ endpoints so each
    tenant gets their own `rate_limit_per_minute` from the DB column.
    Falls back to a conservative 10/min when unauthenticated (blocks abuse
    of endpoints that somehow skipped the auth dep).
    """
    rate = getattr(request.state, "tenant_rate_limit_per_minute", None)
    return f"{rate or 10}/minute"

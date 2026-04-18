"""slowapi limiter configured for tenant-aware keys.

Rate-limiting key order:
    1. request.state.tenant_id — populated by auth when the dep runs
    2. X-Forwarded-For (first hop) when behind Caddy
    3. Peer IP

Every tenant gets their own bucket via `tenant_or_ip` (the key_func), so
tenants don't share budgets. The limit *value* itself is currently a fixed
string per route (see V1_DEFAULT_LIMIT) — slowapi's `@limit` decorator
evaluates its limit string without request context, so truly per-tenant
limit numbers need a custom dependency. TODO: read each tenant's
`rate_limit_per_minute` column in a pre-route dep and short-circuit with
429 when the tenant's ceiling is below the route default.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

# One default rate for every /v1/* endpoint. Liberal enough for a game
# client's idle chatter; abusive traffic still gets cut off. Raise this
# when we add per-tenant overrides.
V1_DEFAULT_LIMIT = "60/minute"


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

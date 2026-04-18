"""X-API-Key authentication for /v1/* endpoints.

The header value is hashed (SHA-256) and looked up against `api_keys.key_hash`.
Successful lookups are cached for `tenant_cache_ttl_seconds` in a process-
local TTL cache so a busy game client doesn't hammer Postgres with auth
queries. The 60-second default means revocations propagate within ~1 min.

Design notes:
    - Raw keys are NEVER logged. Only the 8-char `key_prefix` shows up in
      logs / errors when we need to identify a key during triage.
    - The lookup is keyed by the SHA-256 hash via an indexed equality in
      Postgres, so no Python-side comparison of attacker-controllable
      input is needed. If we ever move to a scan-based match, reintroduce
      hmac.compare_digest.
    - The cache is keyed by the hash bytes, not the raw key — even if
      somebody instrumented the cache object, the raw key wouldn't leak.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cachetools import TTLCache
from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from errors import APIError
from models import ApiKey, Tenant
from settings import settings


@dataclass(frozen=True)
class TenantIdentity:
    """Immutable snapshot of the authenticated tenant for the current request."""

    tenant_id: str
    tenant_slug: str
    rate_limit_per_minute: int
    allowed_origins: tuple[str, ...]
    api_key_id: str
    key_prefix: str


def hash_key(raw_key: str) -> bytes:
    """SHA-256 of a raw API key. 32 bytes."""
    return hashlib.sha256(raw_key.encode("utf-8")).digest()


_cache: TTLCache[bytes, TenantIdentity] = TTLCache(
    maxsize=1024,
    ttl=settings.tenant_cache_ttl_seconds,
)


def _clear_cache_for_tests() -> None:
    """Used by tests to reset between cases. Not exported on the prod surface."""
    _cache.clear()


# Documented security scheme. Shows up in the generated OpenAPI as
# `components.securitySchemes.ApiKeyAuth` and gets applied to every
# endpoint that depends on `get_current_tenant`. `auto_error=False`
# lets our handler produce the 401 envelope instead of FastAPI's default
# 403 {detail: "Not authenticated"}.
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _resolve(raw_key: str, db: AsyncSession) -> TenantIdentity:
    key_hash = hash_key(raw_key)

    cached = _cache.get(key_hash)
    if cached is not None:
        return cached

    stmt = (
        select(ApiKey, Tenant)
        .join(Tenant, Tenant.id == ApiKey.tenant_id)
        .where(ApiKey.key_hash == key_hash)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        raise APIError(401, "unauthenticated", "Invalid API key")

    api_key: ApiKey = row[0]
    tenant: Tenant = row[1]

    if api_key.revoked_at is not None:
        raise APIError(403, "forbidden", "API key has been revoked")

    if tenant.status != "active":
        raise APIError(403, "forbidden", f"Tenant is {tenant.status}")

    identity = TenantIdentity(
        tenant_id=str(tenant.id),
        tenant_slug=tenant.slug,
        rate_limit_per_minute=tenant.rate_limit_per_minute,
        allowed_origins=tuple(tenant.allowed_origins),
        api_key_id=str(api_key.id),
        key_prefix=api_key.key_prefix,
    )
    _cache[key_hash] = identity
    return identity


async def get_current_tenant(
    request: Request,
    x_api_key: str | None = Depends(_api_key_scheme),
    db: AsyncSession = Depends(get_db),
) -> TenantIdentity:
    """FastAPI dependency — resolve X-API-Key to a TenantIdentity.

    Also populates `request.state.tenant_id` and
    `request.state.tenant_rate_limit_per_minute` so the slowapi limiter
    (running further down the middleware chain) can key on the tenant.

    Raises:
        APIError(401) — missing / malformed / unknown key.
        APIError(403) — key revoked or tenant suspended.
    """
    if not x_api_key or len(x_api_key) < 8:
        raise APIError(401, "unauthenticated", "Missing or malformed X-API-Key header")
    identity = await _resolve(x_api_key, db)
    request.state.tenant_id = identity.tenant_id
    request.state.tenant_slug = identity.tenant_slug
    request.state.tenant_rate_limit_per_minute = identity.rate_limit_per_minute
    request.state.api_key_id = identity.api_key_id
    return identity


async def enforce_tenant_origin(
    request: Request,
    identity: TenantIdentity = Depends(get_current_tenant),
) -> TenantIdentity:
    """Per-tenant Origin allowlist check.

    Browser-initiated requests carry an `Origin` header — we match it
    against the tenant's `allowed_origins` list and reject mismatches with
    403. Direct HTTP clients (game servers, CLIs, server-to-server calls)
    omit `Origin`; those pass through. Tenants with an empty
    `allowed_origins` opt out of browser enforcement entirely — common for
    server-only integrations.

    This complements the global CORS middleware, which is permissive by
    design (browsers strip X-API-Key from preflight, so real per-tenant
    enforcement has to happen after auth). An attacker that ignores the
    browser's CORS response and makes a direct HTTP call would still be
    stopped here.

    Raises:
        APIError(403) — Origin header present and not in allowed_origins.
    """
    origin = request.headers.get("Origin")
    if not origin or not identity.allowed_origins:
        return identity
    if origin not in identity.allowed_origins:
        raise APIError(
            403,
            "forbidden",
            f"Origin {origin!r} is not allowed for this tenant",
        )
    return identity

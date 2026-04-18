"""X-API-Key authentication for /v1/* endpoints.

The header value is hashed (SHA-256) and looked up against `api_keys.key_hash`.
Successful lookups are cached for `tenant_cache_ttl_seconds` in a process-
local TTL cache so a busy game client doesn't hammer Postgres with auth
queries. The 60-second default means revocations propagate within ~1 min.

Design notes:
    - Raw keys are NEVER logged. Only the 8-char `key_prefix` shows up in
      logs / errors when we need to identify a key during triage.
    - Comparisons use hmac.compare_digest where the hash is attacker-
      controllable, to avoid timing side-channels even though the lookup
      is already keyed by the hash (the constant-time check here is
      belt-and-braces).
    - The cache is keyed by the hash bytes, not the raw key — even if
      somebody instrumented the cache object, the raw key wouldn't leak.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from cachetools import TTLCache
from fastapi import Depends, Header, Request
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

    if not hmac.compare_digest(bytes(api_key.key_hash), key_hash):
        raise APIError(401, "unauthenticated", "Invalid API key")

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
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
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

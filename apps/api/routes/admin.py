"""Admin API — tenant + api-key CRUD.

Gated by a master `X-Admin-Key` that is NOT a tenant key. There is only
one admin key per process (read from ADMIN_API_KEY env); when it's
empty, every route returns 503 so operators who prefer out-of-band
provisioning can disable the surface.

Returned tenant + api-key shapes deliberately match the internal ORM
fields 1:1 so the surface stays trivial to reason about.

An IP-based rate limit (`enforce_admin_ip_rate_limit`) runs on every
route as a router-level dependency — defense-in-depth against online
brute-force of `ADMIN_API_KEY`. See `rate_limit.py` and the
`ADMIN_IP_RATE_PER_MINUTE` setting. The cap is generous enough for a
human operator; raise it via env if you run scripted admin flows from
a shared egress IP.

Key issuance notes:
    - The raw key is generated server-side (tk_<32-byte hex>) so clients
      can't force a predictable value.
    - The raw value appears in the response exactly ONCE. We hash and
      discard immediately afterward. Future retrieval is impossible.
    - key_prefix (first 8 chars) is stored and returned by list ops so
      operators can identify a key during rotation without the raw value.
"""

from __future__ import annotations

import datetime
import hashlib
import secrets
import uuid

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from errors import APIError
from models import ApiKey, Tenant
from rate_limit import enforce_admin_ip_rate_limit
from settings import settings

# IP-based rate limit runs *before* the admin-key check on every route —
# caps online brute-force of X-Admin-Key. Admin-key check is still the
# real boundary; rate limit is defense-in-depth.
router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(enforce_admin_ip_rate_limit)],
)


# ─── Schemas ──────────────────────────────────────────────────────────────


class TenantCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str = Field(..., min_length=1, max_length=128)
    rate_limit_per_minute: int = Field(60, ge=1, le=100_000)
    allowed_origins: list[str] = Field(default_factory=list)


class TenantUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    rate_limit_per_minute: int | None = Field(default=None, ge=1, le=100_000)
    allowed_origins: list[str] | None = None
    status: str | None = Field(default=None, pattern=r"^(active|suspended)$")


class TenantOut(BaseModel):
    id: str
    slug: str
    name: str
    rate_limit_per_minute: int
    allowed_origins: list[str]
    status: str
    created_at: str
    updated_at: str


class TenantListOut(BaseModel):
    tenants: list[TenantOut]
    count: int


class ApiKeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field("default", min_length=1, max_length=64)


class ApiKeyIssued(BaseModel):
    id: str
    tenant_id: str
    label: str
    key_prefix: str
    created_at: str
    # Raw value — returned exactly once at creation time. Never stored,
    # never retrievable again.
    raw_key: str


class ApiKeyOut(BaseModel):
    id: str
    tenant_id: str
    label: str
    key_prefix: str
    created_at: str
    revoked_at: str | None


class ApiKeyListOut(BaseModel):
    api_keys: list[ApiKeyOut]
    count: int


# ─── Auth ─────────────────────────────────────────────────────────────────


def _require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    """Gate — same shape as the internal-token guard.

    Unconfigured admin_api_key disables the surface (404-via-503 is
    confusing; we surface as 503 to make misconfig loud instead of
    silently 401-ing every call)."""
    if not settings.admin_api_key:
        raise APIError(503, "agent_unavailable", "Admin API is not configured")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise APIError(401, "unauthenticated", "Invalid admin key")


# ─── Helpers ──────────────────────────────────────────────────────────────


def _to_out(t: Tenant) -> TenantOut:
    return TenantOut(
        id=str(t.id),
        slug=t.slug,
        name=t.name,
        rate_limit_per_minute=t.rate_limit_per_minute,
        allowed_origins=list(t.allowed_origins),
        status=t.status,
        created_at=t.created_at.isoformat(),
        updated_at=t.updated_at.isoformat(),
    )


def _key_to_out(k: ApiKey) -> ApiKeyOut:
    return ApiKeyOut(
        id=str(k.id),
        tenant_id=str(k.tenant_id),
        label=k.label,
        key_prefix=k.key_prefix,
        created_at=k.created_at.isoformat(),
        revoked_at=k.revoked_at.isoformat() if k.revoked_at else None,
    )


# ─── Tenants ──────────────────────────────────────────────────────────────


@router.get("", summary="Check admin auth", include_in_schema=False)
async def admin_probe(_: None = Depends(_require_admin)) -> dict[str, str]:
    """Lightweight 200-OK probe — lets operators verify their key works
    without inspecting side effects."""
    return {"status": "ok"}


@router.post(
    "/tenants",
    response_model=TenantOut,
    status_code=201,
    summary="Create a tenant",
)
async def create_tenant(
    body: TenantCreate,
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantOut:
    # Unique slug — pre-check for a friendly 409 rather than falling
    # through to a bare IntegrityError.
    existing = (
        await db.execute(select(Tenant).where(Tenant.slug == body.slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise APIError(409, "conflict", f"Tenant slug {body.slug!r} already exists")

    tenant = Tenant(
        slug=body.slug,
        name=body.name,
        rate_limit_per_minute=body.rate_limit_per_minute,
        allowed_origins=body.allowed_origins,
    )
    db.add(tenant)
    await db.flush()
    await db.refresh(tenant)
    await db.commit()
    return _to_out(tenant)


@router.get(
    "/tenants",
    response_model=TenantListOut,
    summary="List all tenants",
)
async def list_tenants(
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantListOut:
    rows = (await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))).scalars().all()
    return TenantListOut(tenants=[_to_out(t) for t in rows], count=len(rows))


@router.patch(
    "/tenants/{slug}",
    response_model=TenantOut,
    summary="Update a tenant (partial)",
)
async def update_tenant(
    slug: str,
    body: TenantUpdate,
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantOut:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if tenant is None:
        raise APIError(404, "not_found", f"Tenant {slug!r} not found")

    if body.name is not None:
        tenant.name = body.name
    if body.rate_limit_per_minute is not None:
        tenant.rate_limit_per_minute = body.rate_limit_per_minute
    if body.allowed_origins is not None:
        tenant.allowed_origins = body.allowed_origins
    if body.status is not None:
        tenant.status = body.status

    await db.flush()
    await db.refresh(tenant)
    await db.commit()
    return _to_out(tenant)


# ─── API keys ─────────────────────────────────────────────────────────────


def _generate_raw_key() -> str:
    """Produce a fresh `tk_<32-byte hex>` key. 256 bits of entropy."""
    return "tk_" + secrets.token_hex(32)


@router.post(
    "/tenants/{slug}/api-keys",
    response_model=ApiKeyIssued,
    status_code=201,
    summary="Issue a new API key for a tenant (raw value returned once)",
)
async def issue_api_key(
    slug: str,
    body: ApiKeyCreate,
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyIssued:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if tenant is None:
        raise APIError(404, "not_found", f"Tenant {slug!r} not found")

    raw = _generate_raw_key()
    api_key = ApiKey(
        tenant_id=tenant.id,
        key_hash=hashlib.sha256(raw.encode("utf-8")).digest(),
        key_prefix=raw[:8],
        label=body.label,
    )
    db.add(api_key)
    await db.flush()
    await db.refresh(api_key)
    await db.commit()

    return ApiKeyIssued(
        id=str(api_key.id),
        tenant_id=str(api_key.tenant_id),
        label=api_key.label,
        key_prefix=api_key.key_prefix,
        created_at=api_key.created_at.isoformat(),
        raw_key=raw,
    )


@router.get(
    "/tenants/{slug}/api-keys",
    response_model=ApiKeyListOut,
    summary="List API keys for a tenant (metadata only; no raw values)",
)
async def list_api_keys(
    slug: str,
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyListOut:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if tenant is None:
        raise APIError(404, "not_found", f"Tenant {slug!r} not found")

    rows = (
        (
            await db.execute(
                select(ApiKey)
                .where(ApiKey.tenant_id == tenant.id)
                .order_by(ApiKey.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return ApiKeyListOut(api_keys=[_key_to_out(k) for k in rows], count=len(rows))


@router.post(
    "/api-keys/{key_id}/revoke",
    response_model=ApiKeyOut,
    summary="Revoke an API key",
    description=(
        "Sets `revoked_at = now()`. Subsequent requests with this key get "
        "403 forbidden. Idempotent: re-revoking preserves the original "
        "revocation timestamp."
    ),
)
async def revoke_api_key(
    key_id: str,
    _: None = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyOut:
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise APIError(404, "not_found", "API key not found")

    api_key = (await db.execute(select(ApiKey).where(ApiKey.id == key_uuid))).scalar_one_or_none()
    if api_key is None:
        raise APIError(404, "not_found", "API key not found")

    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.datetime.now(datetime.UTC)
        await db.flush()
        await db.refresh(api_key)
        await db.commit()

    return _key_to_out(api_key)

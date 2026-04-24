"""Tenant lookup — the gate on every retriever entry point.

`resolve_tenant(tenant_id)` confirms the tenant_id is a known, active row
in `tenants` before any retrieval work happens. Unknown, suspended, or
disabled tenants all raise `UnknownTenant(400)` with the same fixed
message — the distinction between "doesn't exist" and "exists but
suspended" is NOT surfaced to callers (caller-side tenant-existence
oracle mitigation; constitution Principle IV). The full reason is logged
server-side as structured context so on-call can still debug.

v1 performs no caching — the `tenants` table is tiny, the query hits an
indexed primary key, and a cache would share state across tenants
(exactly the footgun the `xff-spoof` learning warns against). If tenant
lookup ever becomes a measurable cost, add a TTLCache here and keep it
isolated from `apps/api/rate_limit.py`'s caches.

The `_ACTIVE_STATUS` constant is the single source of truth within this
service for the "usable tenant" state. `apps/api` ultimately owns the
enum; drift between the two services (new states like `pending`, `trial`)
would silently fail closed here — safe by default, but a follow-up ticket
tracks turning this into a shared enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from errors import UnknownTenant

_ACTIVE_STATUS = "active"
_UNKNOWN_TENANT_MESSAGE = "unknown tenant"


@dataclass(frozen=True)
class Tenant:
    """Minimal tenant view needed by the retriever."""

    id: UUID
    slug: str
    status: str


async def resolve_tenant(session: AsyncSession, tenant_id: UUID) -> Tenant:
    """Look up a tenant by id. Raises UnknownTenant if missing or non-active.

    Both cases return the same `UnknownTenant` error with the same fixed
    message so a caller can't tell the difference (existence oracle /
    suspended-status oracle). Server-side logs carry the actual reason.
    """
    row = (
        (
            await session.execute(
                text("SELECT id, slug, status FROM tenants WHERE id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
        )
        .mappings()
        .first()
    )

    if row is None:
        logger.info("tenant.lookup_missing tenant_id={tenant_id}", tenant_id=tenant_id)
        raise UnknownTenant(_UNKNOWN_TENANT_MESSAGE)

    if row["status"] != _ACTIVE_STATUS:
        logger.warning(
            "tenant.lookup_non_active tenant_id={tenant_id} status={status}",
            tenant_id=tenant_id,
            status=row["status"],
        )
        raise UnknownTenant(_UNKNOWN_TENANT_MESSAGE)

    return Tenant(id=row["id"], slug=row["slug"], status=row["status"])

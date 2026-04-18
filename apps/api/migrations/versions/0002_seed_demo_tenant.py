"""seed: demo tenant + optional API key

Revision ID: 0002_seed_demo_tenant
Revises: 0001_init
Create Date: 2026-04-17 00:00:01

Creates the single 'demo' tenant used by the public web client at
coastalai.ai.

The API key is sourced from SEED_DEMO_API_KEY. When unset, the tenant
is still created but has no key attached — provision later via SQL.
This keeps the migration idempotent across environments: your laptop,
CI, and prod each get their own key without baking a shared secret
into this file.

Infrastructure generates + supplies the key via Terraform and threads
it into the api-migrate container's env.
"""

from __future__ import annotations

import hashlib
import os

import sqlalchemy as sa
from alembic import op

revision = "0002_seed_demo_tenant"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def _hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _prefix(raw: str) -> str:
    return raw[:8]


def upgrade() -> None:
    bind = op.get_bind()
    # ON CONFLICT makes re-runs safe even though Alembic wouldn't normally
    # re-execute a migration — defensive against someone running this SQL
    # directly.
    bind.execute(
        sa.text(
            """
            INSERT INTO tenants (slug, name, rate_limit_per_minute, allowed_origins)
            VALUES (:slug, :name, :rate_limit, :origins)
            ON CONFLICT (slug) DO UPDATE
                SET rate_limit_per_minute = EXCLUDED.rate_limit_per_minute,
                    allowed_origins = EXCLUDED.allowed_origins
            """
        ),
        {
            "slug": "demo",
            "name": "Public Demo",
            "rate_limit": 10,
            "origins": ["https://coastalai.ai"],
        },
    )

    raw = os.environ.get("SEED_DEMO_API_KEY", "").strip()
    if not raw:
        # No key provided — tenant exists but is unauthenticated. Operator
        # can insert one later via SQL. Common for first-run CI where keys
        # aren't set yet.
        return
    if len(raw) < 16:
        raise RuntimeError("SEED_DEMO_API_KEY must be at least 16 characters")

    bind.execute(
        sa.text(
            """
            INSERT INTO api_keys (tenant_id, key_hash, key_prefix, label)
            SELECT t.id, :hash, :prefix, :label FROM tenants t WHERE t.slug = :slug
            ON CONFLICT (key_hash) DO NOTHING
            """
        ),
        {
            "hash": _hash(raw),
            "prefix": _prefix(raw),
            "label": "seed",
            "slug": "demo",
        },
    )


def downgrade() -> None:
    op.execute("DELETE FROM tenants WHERE slug = 'demo'")

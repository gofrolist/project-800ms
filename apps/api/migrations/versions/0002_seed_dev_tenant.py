"""seed: dev + demo tenants with API keys

Revision ID: 0002_seed_dev_tenant
Revises: 0001_init
Create Date: 2026-04-17 00:00:01

Creates two tenants on a fresh DB:
    - 'dev'  — localhost development, 60 req/min
    - 'demo' — public web client at coastalai.ai, 10 req/min

API keys are sourced from env vars (SEED_DEV_API_KEY, SEED_DEMO_API_KEY).
When either is unset, the corresponding tenant is still created but has no
key attached — provision later via SQL. This keeps the migration idempotent
across environments: your laptop, CI, and prod each get their own keys
without baking a shared secret into this file.

Infrastructure generates + supplies these via Terraform (random_password)
and threads them into the api-migrate container's env.
"""

from __future__ import annotations

import hashlib
import os

import sqlalchemy as sa
from alembic import op

revision = "0002_seed_dev_tenant"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def _hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _prefix(raw: str) -> str:
    return raw[:8]


def _upsert_tenant_with_key(
    *,
    slug: str,
    name: str,
    rate_limit: int,
    allowed_origins: list[str],
    raw_key_env: str,
) -> None:
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
            "slug": slug,
            "name": name,
            "rate_limit": rate_limit,
            "origins": allowed_origins,
        },
    )
    raw = os.environ.get(raw_key_env, "").strip()
    if not raw:
        # No key provided — tenant exists but is unauthenticated. Operator
        # can insert one later. Common for first-run CI where keys aren't
        # set yet.
        return
    if len(raw) < 16:
        raise RuntimeError(f"{raw_key_env} must be at least 16 characters")
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
            "slug": slug,
        },
    )


def upgrade() -> None:
    _upsert_tenant_with_key(
        slug="dev",
        name="Local Development",
        rate_limit=60,
        allowed_origins=["http://localhost:5173"],
        raw_key_env="SEED_DEV_API_KEY",
    )
    _upsert_tenant_with_key(
        slug="demo",
        name="Public Demo",
        rate_limit=10,
        allowed_origins=["https://coastalai.ai"],
        raw_key_env="SEED_DEMO_API_KEY",
    )


def downgrade() -> None:
    op.execute("DELETE FROM tenants WHERE slug IN ('dev', 'demo')")

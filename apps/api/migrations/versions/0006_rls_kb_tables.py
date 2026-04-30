"""Row-level security on kb_entries / kb_chunks / retrieval_traces.

Revision ID: 0006_rls_kb_tables
Revises: 0005_traces_restrict
Create Date: 2026-04-27 00:01:00

Issue #41: tenant isolation today is enforced at the SQL layer via
``WHERE tenant_id = $1`` predicates in every query. A single forgotten
predicate silently leaks data across tenants — there is no type-level
or DB-level guard. Constitution Principle IV calls tenant isolation
"a data invariant, not a convention" — this migration moves it to a
data invariant.

Mechanism: Postgres row-level security with per-transaction GUCs.

  1. Enable row-level security on the three tenant-scoped tables.
     One policy per table: rows match iff
     ``tenant_id = current_setting('app.current_tenant_id', true)::uuid``.
     The ``COALESCE`` to the zero UUID makes "GUC unset" → "no rows
     match" (fail-closed). USING applies to SELECT/UPDATE/DELETE,
     WITH CHECK applies to INSERT/UPDATE — neither path can leak.

  2. ``FORCE ROW LEVEL SECURITY`` is intentionally NOT applied. RLS
     in Postgres bypasses for the table OWNER unless FORCE is set;
     the local dev / test environment runs as the `voice` superuser
     which IS the owner, so without FORCE the policies are
     effectively a no-op for tests and migrations. Production
     deployments MUST connect as a non-superuser, non-owner role
     (e.g. `voice_app`) for the policies to actually fire — that
     role rollout is filed as a follow-up. Until then RLS is
     defense-in-depth ALONGSIDE the WHERE-clause discipline, not
     instead of it.

Calling-side contract (retriever, agent, API): every code path that
touches these tables MUST first run

    SELECT set_config('app.current_tenant_id', '<uuid>', true);

inside its transaction. ``true`` is the ``is_local`` flag — the GUC
auto-resets at transaction commit, so it cannot leak across requests
sharing a connection from the pool. The retriever exposes
``services/retriever/db.py::set_tenant_scope`` for this; the
in-process retrieve handler calls it after ``resolve_tenant``.

Up: enable RLS on the three tables; create the policies.

Down: drop the policies; disable RLS. Schema is identical pre/post
either way — only the guard is added/removed.
"""

from __future__ import annotations

from alembic import op


revision = "0006_rls_kb_tables"
down_revision = "0005_traces_restrict"
branch_labels = None
depends_on = None


_TABLES = ("kb_entries", "kb_chunks", "retrieval_traces")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # COALESCE → zero UUID guarantees "GUC unset" denies all rows.
        # No real tenant ever has the zero UUID (uuid_generate_v4 has
        # nonzero version + variant nibbles), so a fail-closed default
        # is unambiguous.
        op.execute(
            f"""
            CREATE POLICY tenant_isolation_{table}
            ON {table}
            USING (
                tenant_id = COALESCE(
                    current_setting('app.current_tenant_id', true),
                    '00000000-0000-0000-0000-000000000000'
                )::uuid
            )
            WITH CHECK (
                tenant_id = COALESCE(
                    current_setting('app.current_tenant_id', true),
                    '00000000-0000-0000-0000-000000000000'
                )::uuid
            )
            """
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

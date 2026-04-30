"""retrieval_traces.session_id ON DELETE RESTRICT (was CASCADE)

Revision ID: 0005_traces_restrict
Revises: 0004_kb_chunks
Create Date: 2026-04-27 00:00:00

Issue #43: ``retrieval_traces.session_id`` was originally declared
``ON DELETE CASCADE`` (migration 0004), but ``data-model.md`` claims
traces are append-only and immutable once written. Cascade-on-delete
contradicts that — a session retention job that hard-deletes a row
from ``sessions`` would silently wipe all the correlated trace rows,
including refusal audits.

Switching to ``ON DELETE RESTRICT`` forces retention work to delete
trace rows EXPLICITLY before the parent session row, making the
intent visible and auditable. GDPR-style erasure flows can still
achieve full deletion — they just have to walk the trace rows
deliberately.

Trade-off: developers running ``DELETE FROM sessions WHERE ...``
without thinking about traces will hit a constraint error. That's
the point — silent forensic loss is the worse failure mode.

Up: drop the existing FK, recreate it with ``ON DELETE RESTRICT``.
Down: drop the RESTRICT FK, recreate it with the original CASCADE
(non-destructive — both shapes preserve the FK, only the delete
behavior changes).
"""

from __future__ import annotations

from alembic import op


revision = "0005_traces_restrict"
down_revision = "0004_kb_chunks"
branch_labels = None
depends_on = None


_FK_NAME = "retrieval_traces_session_id_fkey"


def upgrade() -> None:
    op.drop_constraint(_FK_NAME, "retrieval_traces", type_="foreignkey")
    op.create_foreign_key(
        _FK_NAME,
        "retrieval_traces",
        "sessions",
        ["session_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(_FK_NAME, "retrieval_traces", type_="foreignkey")
    op.create_foreign_key(
        _FK_NAME,
        "retrieval_traces",
        "sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )

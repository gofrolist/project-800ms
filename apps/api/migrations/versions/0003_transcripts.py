"""session_transcripts table

Revision ID: 0003_transcripts
Revises: 0002_seed_demo_tenant
Create Date: 2026-04-18 12:00:00

One row per final utterance (user STT-completion or assistant LLM-end).
Rows are immutable — no updates, only inserts. Cascade-deleted when the
owning session is deleted so retention policies propagate automatically.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_transcripts"
down_revision = "0002_seed_demo_tenant"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_transcripts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name="session_transcripts_role_check",
        ),
    )
    # Most queries scan one session's transcripts ordered by time — a
    # composite index on (session_id, created_at) makes that a single
    # range scan. FK index on session_id alone comes for free from
    # Postgres's FK machinery but we're explicit.
    op.create_index(
        "ix_session_transcripts_session_created",
        "session_transcripts",
        ["session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_session_transcripts_session_created",
        table_name="session_transcripts",
    )
    op.drop_table("session_transcripts")

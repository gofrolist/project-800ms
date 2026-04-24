"""kb_entries + kb_chunks + retrieval_traces (Helper/Guide NPC RAG)

Revision ID: 0004_kb_chunks
Revises: 0003_transcripts
Create Date: 2026-04-24 00:00:00

Introduces the schema for KB-grounded Helper/Guide NPC answers:

  - `kb_entries`       — source-of-truth unit per tenant KB.
  - `kb_chunks`        — retrievable unit; pgvector HNSW + Russian GIN on
                         a GENERATED tsvector. Supports content chunks and
                         synthetic-question chunks in the same table,
                         discriminated by `is_synthetic_question` with a
                         self-FK `parent_chunk_id`.
  - `retrieval_traces` — append-only per-turn forensics substrate.

All entities are tenant-scoped. Every retrieval query MUST carry
`WHERE tenant_id = $1` (constitution Principle IV; spec 002 FR-002).

The `vector` extension is provided by the `pgvector/pgvector:pg18` base
image (see infra/docker-compose.yml). It is NOT bundled with upstream
Postgres — running this migration against stock `postgres:*` images will
fail at `CREATE EXTENSION vector`. `pg_trgm` is a standard Postgres
contrib extension bundled with upstream Postgres. Both statements are
`CREATE ... IF NOT EXISTS` and do not touch any existing table, so this
migration is safe on a live system.

See specs/002-helper-guide-npc/data-model.md for field-level contracts.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_kb_chunks"
down_revision = "0003_transcripts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ──────────────────────────────────────────────────────
    # pgvector: vector(n) column type + HNSW index support.
    # pg_trgm: trigram similarity — kept as an escape hatch for BM25-ish
    # lexical fallback; not load-bearing for hybrid retrieval which uses
    # to_tsvector('russian', ...) directly.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ── kb_entries ──────────────────────────────────────────────────────
    # Source units. One row per article / page in the tenant's source KB.
    # kb_entry_key is the tenant's own stable identifier (e.g. article
    # slug, wiki page id). Uniqueness is (tenant_id, kb_entry_key), not
    # tenant-global kb_entry_key — two tenants MAY share slugs.
    op.create_table(
        "kb_entries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kb_entry_key", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "kb_entry_key",
            name="uq_kb_entries_tenant_key",
        ),
    )
    op.create_index("ix_kb_entries_tenant", "kb_entries", ["tenant_id"])

    # ── kb_chunks ───────────────────────────────────────────────────────
    # Retrievable units. Two row species in one table:
    #   1. Content chunks:     is_synthetic_question=false, parent_chunk_id=NULL
    #   2. Synthetic questions: is_synthetic_question=true,  parent_chunk_id → a content chunk
    # embedding dimensionality is fixed at 1024 (BGE-M3; research R1).
    #
    # content_tsv is GENERATED, not maintained by triggers, to eliminate the
    # classic "forgot to update tsvector on UPDATE" footgun.
    op.execute(
        """
        CREATE TABLE kb_chunks (
            id                    BIGSERIAL PRIMARY KEY,
            tenant_id             UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            kb_entry_id           UUID NOT NULL REFERENCES kb_entries(id) ON DELETE CASCADE,
            section               TEXT NULL,
            title                 TEXT NOT NULL,
            content               TEXT NOT NULL,
            content_tsv           TSVECTOR GENERATED ALWAYS AS (
                                    to_tsvector('russian', title || ' ' || content)
                                  ) STORED,
            content_sha256        TEXT NOT NULL,
            embedding             VECTOR(1024) NOT NULL,
            metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_synthetic_question BOOLEAN NOT NULL DEFAULT FALSE,
            parent_chunk_id       BIGINT NULL REFERENCES kb_chunks(id) ON DELETE CASCADE,
            version               INT NOT NULL DEFAULT 1,
            ingested_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (
                (is_synthetic_question = FALSE AND parent_chunk_id IS NULL)
                OR
                (is_synthetic_question = TRUE AND parent_chunk_id IS NOT NULL)
            )
        )
        """
    )

    # Composite uniqueness. Content chunks collide on
    # (tenant_id, kb_entry_id, section); synthetic questions allow many
    # per parent via the discriminator.
    #
    # NULLS NOT DISTINCT is load-bearing: `section` is nullable (entries
    # without H2/H3 structure), and Postgres's default NULLS DISTINCT
    # would let two content chunks with (tenant, entry, NULL) coexist,
    # defeating the idempotent-upsert design (SC-009). Available since
    # PG 15; our prod target is PG 18.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_kb_chunks_content
            ON kb_chunks (tenant_id, kb_entry_id, section)
            NULLS NOT DISTINCT
            WHERE is_synthetic_question = FALSE
        """
    )

    # HNSW semantic index. m=16 / ef_construction=200 per research R4.
    op.execute(
        """
        CREATE INDEX ix_kb_chunks_embedding
            ON kb_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)
        """
    )

    # Lexical index on the generated tsvector.
    op.execute("CREATE INDEX ix_kb_chunks_tsv ON kb_chunks USING gin (content_tsv)")

    # Tenant- and entry-scoped scans (bulk re-ingest, per-entry counts).
    op.create_index(
        "ix_kb_chunks_tenant_entry",
        "kb_chunks",
        ["tenant_id", "kb_entry_id"],
    )
    op.create_index("ix_kb_chunks_tenant", "kb_chunks", ["tenant_id"])

    # ── retrieval_traces ────────────────────────────────────────────────
    # Append-only per-turn record. Immutability enforced by code — this
    # migration creates the table without any UPDATE trigger, and the
    # retriever service performs INSERT-only writes (see T026).
    op.create_table(
        "retrieval_traces",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("raw_transcript", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("in_scope", sa.Boolean(), nullable=True),
        sa.Column("rewriter_version", sa.Text(), nullable=False),
        sa.Column(
            "retrieved_chunks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "stage_timings_ms",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("final_reply_text", sa.Text(), nullable=True),
        sa.Column("error_class", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "session_id",
            "turn_id",
            name="uq_retrieval_traces_session_turn",
        ),
    )

    # "What did tenant X do in the last hour" — the dominant operator query.
    op.create_index(
        "ix_retrieval_traces_tenant_created",
        "retrieval_traces",
        ["tenant_id", sa.text("created_at DESC")],
    )
    # Session reconstruction ("give me every turn for session S").
    op.create_index(
        "ix_retrieval_traces_session_turn",
        "retrieval_traces",
        ["session_id", "turn_id"],
    )
    # Refusal forensics — partial index keeps it small.
    op.execute(
        """
        CREATE INDEX ix_retrieval_traces_refusals
            ON retrieval_traces (tenant_id, created_at DESC)
            WHERE in_scope = FALSE
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_retrieval_traces_refusals",
        table_name="retrieval_traces",
    )
    op.drop_index(
        "ix_retrieval_traces_session_turn",
        table_name="retrieval_traces",
    )
    op.drop_index(
        "ix_retrieval_traces_tenant_created",
        table_name="retrieval_traces",
    )
    op.drop_table("retrieval_traces")

    op.drop_index("ix_kb_chunks_tenant", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_tenant_entry", table_name="kb_chunks")
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_tsv")
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS uq_kb_chunks_content")
    op.drop_table("kb_chunks")

    op.drop_index("ix_kb_entries_tenant", table_name="kb_entries")
    op.drop_table("kb_entries")

    # Do not drop the extensions in downgrade — other tables / future
    # migrations may rely on them, and DROP EXTENSION is destructive
    # beyond this migration's scope.

"""SQLAlchemy 2.0 ORM models for the retriever.

Mirrors the tables created by `apps/api/migrations/versions/0004_kb_chunks.py`
at the column level. Retrieval queries use raw SQL (see `hybrid_search.py`)
rather than the ORM because the two-CTE hybrid fusion is hand-tuned; these
models exist for trace writes, test fixture seeding, and type-safe access
in scripts.

Deliberate omissions:

* `kb_chunks.content_tsv` — GENERATED ALWAYS column in the migration.
  Never written from Python; read via raw SQL in the lexical CTE. Omitting
  it from the ORM is simpler than teaching SQLAlchemy to treat it as
  `Computed` (which is only metadata for schema-generation tools anyway;
  migrations own the DDL here).
* Partial + HNSW + GIN indexes (`uq_kb_chunks_content`, `ix_kb_chunks_embedding`,
  `ix_kb_chunks_tsv`, `ix_retrieval_traces_refusals`). These exist in the
  migration and are visible to the query planner; re-declaring them here
  would only be useful if the ORM owned schema generation.

The `is_synthetic_question` + `parent_chunk_id` columns on KBChunk are
present but unused in US1 — populated in US4 ingestion (synthetic
questions) and returned alongside content chunks by the same hybrid
query.

Every query that touches these tables MUST carry an explicit
`WHERE tenant_id = $1` (constitution Principle IV). There is NO
type-level enforcement; a repository layer is tracked as Phase 3+
follow-up (see `services/retriever/db.py` docstring).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Retriever-local declarative base.

    Deliberately NOT shared with `apps/api/models.py::Base` — the two
    services live in the same DB but own independent ORM layers. Sharing
    a Base would require a cross-service import and couple deployment
    lifecycles.

    Consequence: FK columns pointing at `tenants.id` and `sessions.id`
    (owned by `apps/api`) CANNOT carry a SQLAlchemy `ForeignKey()` here
    — SQLAlchemy would fail to resolve the target table against this
    local metadata at flush time. The DB-level FK constraints created
    by migration 0004 still fire (including ON DELETE CASCADE), so no
    integrity guarantee is lost; we just don't redeclare them ORM-side.
    """


class KBEntry(Base):
    """Source-of-truth unit per tenant KB (one row per KB article/page)."""

    __tablename__ = "kb_entries"
    __table_args__ = (
        UniqueConstraint("tenant_id", "kb_entry_key", name="uq_kb_entries_tenant_key"),
        Index("ix_kb_entries_tenant", "tenant_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sql_text("gen_random_uuid()"),
    )
    # FK to tenants.id enforced at DB level by migration 0004 — see
    # Base docstring for why this ORM column omits the ForeignKey().
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    kb_entry_key: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sql_text("now()"),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sql_text("now()"),
    )

    chunks: Mapped[list[KBChunk]] = relationship(
        back_populates="entry",
        cascade="all, delete-orphan",
    )


class KBChunk(Base):
    """Retrievable unit — content chunks + synthetic-question chunks in one table."""

    __tablename__ = "kb_chunks"
    __table_args__ = (
        CheckConstraint(
            "(is_synthetic_question = FALSE AND parent_chunk_id IS NULL) OR "
            "(is_synthetic_question = TRUE AND parent_chunk_id IS NOT NULL)",
            name="kb_chunks_synthetic_invariant_check",
        ),
        Index("ix_kb_chunks_tenant_entry", "tenant_id", "kb_entry_id"),
        Index("ix_kb_chunks_tenant", "tenant_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # FK to tenants.id enforced at DB level (see Base docstring).
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    kb_entry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("kb_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # content_tsv is GENERATED ALWAYS in the migration — intentionally
    # not declared here (see module docstring).
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    # `metadata` is a reserved attribute on DeclarativeBase (holds the
    # MetaData registry) — map the Python attribute `extra` to the DB
    # column `metadata` instead of shadowing it.
    extra: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=sql_text("'{}'::jsonb"),
    )
    is_synthetic_question: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=sql_text("FALSE"),
    )
    parent_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("kb_chunks.id", ondelete="CASCADE"),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("1"))
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sql_text("now()"),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sql_text("now()"),
    )

    entry: Mapped[KBEntry] = relationship(
        back_populates="chunks",
        foreign_keys=[kb_entry_id],
    )
    parent: Mapped[KBChunk | None] = relationship(
        remote_side=[id],
        foreign_keys=[parent_chunk_id],
    )


class RetrievalTrace(Base):
    """Append-only per-turn record.

    Immutability is enforced at the code level — the retriever never
    issues UPDATE against this table. Any attempt to modify an existing
    row indicates a bug (guarded by T059 in US5). The migration creates
    no UPDATE trigger and the service performs INSERT-only writes.
    """

    __tablename__ = "retrieval_traces"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_id", name="uq_retrieval_traces_session_turn"),
        Index(
            "ix_retrieval_traces_tenant_created",
            "tenant_id",
            sql_text("created_at DESC"),
        ),
        Index("ix_retrieval_traces_session_turn", "session_id", "turn_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sql_text("gen_random_uuid()"),
    )
    # FKs to tenants.id and sessions.id enforced at DB level (see Base
    # docstring).
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    turn_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sql_text("now()"),
    )
    raw_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    in_scope: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rewriter_version: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sql_text("'[]'::jsonb"),
    )
    stage_timings_ms: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sql_text("'{}'::jsonb"),
    )
    final_reply_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_class: Mapped[str | None] = mapped_column(Text, nullable=True)

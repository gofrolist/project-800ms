"""SQLAlchemy 2.0 ORM models — tenants, API keys, sessions.

Schema follows the Phase 1 PRD. Primary keys are server-generated UUIDs
(via pgcrypto's gen_random_uuid) so callers never invent IDs. Times are
TIMESTAMPTZ with default now().

Deliberately missing from Phase 1:
    - session_transcripts (Phase 2)
    - audit_logs (Phase 2 — request_id in loguru is enough for now)
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base — all ORM models inherit from this."""


class Tenant(Base):
    __tablename__ = "tenants"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'suspended')", name="tenants_status_check"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    rate_limit_per_minute: Mapped[int] = mapped_column(nullable=False, server_default=text("60"))
    allowed_origins: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'active'"))
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False,
        server_default=text("now()"),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False,
        server_default=text("now()"),
    )

    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_tenant_id", "tenant_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    # SHA-256 of the raw key (32 bytes). Raw key is shown once at creation
    # time and never stored. Unique so lookup is O(1) and duplicates are
    # impossible by construction.
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False, unique=True)
    # First 8 chars of the raw key — safe to log, used by ops to identify
    # keys without grepping the hash.
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'default'"))
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    revoked_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False,
        server_default=text("now()"),
    )

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'active', 'ended', 'failed')",
            name="sessions_status_check",
        ),
        Index("ix_sessions_tenant_created", "tenant_id", "created_at"),
        Index("ix_sessions_api_key_id", "api_key_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id"),
        nullable=False,
    )
    room: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    identity: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    npc_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    persona: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    voice: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False,
        server_default=text("now()"),
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    ended_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    audio_seconds: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))

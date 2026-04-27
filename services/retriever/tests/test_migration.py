"""Integration test for migration 0004_kb_chunks.

Asserts that the Alembic migration introduced in this feature applies
cleanly against a fresh `pgvector/pgvector:pg18` container, creates the
expected extensions, tables, and indexes (with the correct pgvector
operator class on the HNSW index), and downgrades cleanly back to
revision 0003.

This is an integration test per constitution Principle II — hits a real
Postgres + pgvector, no mocks. Verification runs against an async
engine (asyncpg) to match the retriever's installed driver — we don't
want to pull psycopg2 just for the test-side verification.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "apps" / "api"


def _alembic(container: PostgresContainer, *args: str) -> None:
    """Run `alembic <args>` against the container's DSN.

    The `apps/api` Alembic env.py imports `apps/api/settings.py`, which
    loads a Pydantic `Settings` at module import. That class has several
    required fields (LIVEKIT_API_*) that alembic doesn't actually use —
    we fill them with syntactically-valid dummies so the import succeeds
    inside the test subprocess.

    `VIRTUAL_ENV` from the caller (the retriever's venv) is stripped so
    `uv run` resolves against `apps/api/.venv` without emitting the
    "VIRTUAL_ENV does not match project environment" warning.

    The DSN is rewritten to the asyncpg driver — apps/api's
    migrations/env.py builds an async engine (`async_engine_from_config`),
    so a bare `postgresql://` URL would trip SQLAlchemy's psycopg2
    default, which isn't installed here.
    """
    dsn = container.get_connection_url().replace("+psycopg2", "+asyncpg")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(
        {
            "DATABASE_URL": dsn,
            "LIVEKIT_API_KEY": env.get("LIVEKIT_API_KEY", "test-livekit-key"),
            "LIVEKIT_API_SECRET": env.get("LIVEKIT_API_SECRET", "x" * 32),
        }
    )
    subprocess.run(
        ["uv", "run", "alembic", *args],
        cwd=_API_DIR,
        env=env,
        check=True,
    )


def _asyncpg_dsn(container: PostgresContainer) -> str:
    """DSN for the test-side async engine — same driver as the retriever uses."""
    return container.get_connection_url().replace("+psycopg2", "+asyncpg")


@pytest.fixture
def fresh_pgvector() -> Iterator[PostgresContainer]:
    """Fresh container per test (no session reuse) — this suite verifies
    migration up/down cycles in isolation."""
    try:
        container = PostgresContainer(
            image="pgvector/pgvector:pg18",
            username="voice",
            password="test",
            dbname="voice",
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker / testcontainers unavailable: {exc}")
    yield container
    container.stop()


async def test_upgrade_creates_extensions_tables_and_indexes(
    fresh_pgvector: PostgresContainer,
) -> None:
    _alembic(fresh_pgvector, "upgrade", "head")

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    try:
        async with engine.connect() as conn:
            # Extensions
            result = await conn.execute(text("SELECT extname FROM pg_extension"))
            extensions = {row[0] for row in result.all()}
            assert {"vector", "pg_trgm", "pgcrypto"}.issubset(extensions)

            # Tables
            result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            )
            tables = {row[0] for row in result.all()}
            assert {"kb_entries", "kb_chunks", "retrieval_traces"}.issubset(tables)

            # Indexes on kb_chunks — all expected by name
            result = await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname='public' AND tablename='kb_chunks'"
                )
            )
            idx_names = {row[0] for row in result.all()}
            assert "ix_kb_chunks_embedding" in idx_names
            assert "ix_kb_chunks_tsv" in idx_names
            assert "ix_kb_chunks_tenant_entry" in idx_names
            assert "ix_kb_chunks_tenant" in idx_names
            assert "uq_kb_chunks_content" in idx_names

            # HNSW operator class is correct — cosine, not L2.
            # (Wrong operator class silently tanks recall without erroring.)
            result = await conn.execute(
                text(
                    """
                    SELECT am.amname AS index_method, op.opcname
                    FROM pg_index i
                    JOIN pg_class c         ON c.oid = i.indexrelid
                    JOIN pg_am am           ON am.oid = c.relam
                    JOIN pg_opclass op      ON op.oid = ANY(i.indclass)
                    WHERE c.relname = 'ix_kb_chunks_embedding'
                    """
                )
            )
            opclass_row = result.mappings().first()
            assert opclass_row is not None
            assert opclass_row["index_method"] == "hnsw"
            assert opclass_row["opcname"] == "vector_cosine_ops"

            # retrieval_traces indexes, including the refusal partial index
            result = await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname='public' AND tablename='retrieval_traces'"
                )
            )
            trace_idx = {row[0] for row in result.all()}
            assert "ix_retrieval_traces_refusals" in trace_idx
            assert "ix_retrieval_traces_tenant_created" in trace_idx
            assert "ix_retrieval_traces_session_turn" in trace_idx
    finally:
        await engine.dispose()


async def test_downgrade_reverses_cleanly(fresh_pgvector: PostgresContainer) -> None:
    _alembic(fresh_pgvector, "upgrade", "head")
    _alembic(fresh_pgvector, "downgrade", "0003_transcripts")

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            )
            tables = {row[0] for row in result.all()}
            assert "kb_chunks" not in tables
            assert "kb_entries" not in tables
            assert "retrieval_traces" not in tables

            # Extensions intentionally NOT dropped by downgrade — other
            # code may rely on them.
            result = await conn.execute(text("SELECT extname FROM pg_extension"))
            exts = {row[0] for row in result.all()}
            assert "vector" in exts
            assert "pg_trgm" in exts
    finally:
        await engine.dispose()


async def test_kb_chunks_check_constraint_enforces_synthetic_question_invariant(
    fresh_pgvector: PostgresContainer,
) -> None:
    """The CHECK constraint guarantees synthetic questions always carry a
    parent_chunk_id and content chunks never do. Both violating legs must
    fail at the database layer, and the valid positive case must succeed.
    """
    _alembic(fresh_pgvector, "upgrade", "head")

    from sqlalchemy.exc import IntegrityError

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    zero_vector = "[" + ",".join(["0"] * 1024) + "]"

    try:
        # --- Leg 1: synthetic question WITHOUT parent_chunk_id → rejected.
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-slug-1') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            with pytest.raises(IntegrityError):
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'T', 'C', 'h',
                             CAST(:emb AS vector), TRUE, NULL)
                        """
                    ),
                    {"t": tenant_id, "e": entry_id, "emb": zero_vector},
                )

        # --- Leg 2: content chunk WITH a non-null parent_chunk_id → rejected.
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-slug-2') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            # First insert a valid content chunk we can reference.
            parent_id = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'T', 'C', 'h',
                             CAST(:emb AS vector), FALSE, NULL)
                        RETURNING id
                        """
                    ),
                    {"t": tenant_id, "e": entry_id, "emb": zero_vector},
                )
            ).scalar_one()
            with pytest.raises(IntegrityError):
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'T2', 'C2', 'h2',
                             CAST(:emb AS vector), FALSE, :parent)
                        """
                    ),
                    {
                        "t": tenant_id,
                        "e": entry_id,
                        "emb": zero_vector,
                        "parent": parent_id,
                    },
                )

        # --- Positive: synthetic question WITH a valid parent → accepted.
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-slug-3') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            parent_id = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'T', 'C', 'h',
                             CAST(:emb AS vector), FALSE, NULL)
                        RETURNING id
                        """
                    ),
                    {"t": tenant_id, "e": entry_id, "emb": zero_vector},
                )
            ).scalar_one()
            child_id = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'TQ', 'Q?', 'h',
                             CAST(:emb AS vector), TRUE, :parent)
                        RETURNING id
                        """
                    ),
                    {
                        "t": tenant_id,
                        "e": entry_id,
                        "emb": zero_vector,
                        "parent": parent_id,
                    },
                )
            ).scalar_one()
            assert child_id != parent_id
    finally:
        await engine.dispose()


async def test_content_tsv_uses_russian_stemmer(
    fresh_pgvector: PostgresContainer,
) -> None:
    """`content_tsv` is GENERATED with `to_tsvector('russian', ...)`. If a
    future edit swaps the config to 'simple', index exists checks would
    still pass but SC-001 recall would tank silently. Assert a known
    Russian stem ends up in the tsvector."""
    _alembic(fresh_pgvector, "upgrade", "head")

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    zero_vector = "[" + ",".join(["0"] * 1024) + "]"
    try:
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-ru') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            await conn.execute(
                text(
                    """
                    INSERT INTO kb_chunks
                        (tenant_id, kb_entry_id, title, content, content_sha256,
                         embedding, is_synthetic_question, parent_chunk_id)
                    VALUES
                        (:t, :e, 'Водительская лицензия',
                         'Как получить водительские права в автошколе',
                         'h', CAST(:emb AS vector), FALSE, NULL)
                    """
                ),
                {"t": tenant_id, "e": entry_id, "emb": zero_vector},
            )
            tsv_text = (
                await conn.execute(
                    text("SELECT content_tsv::text FROM kb_chunks WHERE tenant_id = :t"),
                    {"t": tenant_id},
                )
            ).scalar_one()

    finally:
        await engine.dispose()

    # Russian snowball stems "водительская" → "водительск", "получить" →
    # "получ", "автошколе" → "автошкол". Substring match is enough to
    # prove the 'russian' config is active.
    assert "водительск" in tsv_text
    assert "получ" in tsv_text


async def test_hnsw_index_has_expected_storage_parameters(
    fresh_pgvector: PostgresContainer,
) -> None:
    """Assert HNSW was created with m=16 and ef_construction=200
    (research R4). Dropping the WITH clause would still pass the
    operator-class check, but retrieval recall would degrade silently."""
    _alembic(fresh_pgvector, "upgrade", "head")

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    try:
        async with engine.connect() as conn:
            reloptions = (
                await conn.execute(
                    text(
                        """
                        SELECT c.reloptions
                          FROM pg_class c
                         WHERE c.relname = 'ix_kb_chunks_embedding'
                        """
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    # reloptions is an array of key=value strings.
    assert reloptions is not None
    joined = ",".join(reloptions)
    assert "m=16" in joined
    assert "ef_construction=200" in joined


async def test_vector_dim_1024_enforced_at_db(
    fresh_pgvector: PostgresContainer,
) -> None:
    """Schema claims VECTOR(1024). Inserting a 768-dim vector MUST fail
    at the DB layer so config-drift (model swap without schema update)
    surfaces before INSERT time."""
    _alembic(fresh_pgvector, "upgrade", "head")

    from sqlalchemy.exc import DBAPIError

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    bad_vector = "[" + ",".join(["0"] * 768) + "]"
    try:
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-dim') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            # pgvector raises "expected N dimensions" which asyncpg wraps
            # as asyncpg.DataError; SQLAlchemy surfaces it as DBAPIError.
            # Assert the message to be sure we're catching the right
            # thing rather than some unrelated failure.
            with pytest.raises(DBAPIError, match=r"expected 1024 dimensions"):
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, title, content, content_sha256,
                             embedding, is_synthetic_question, parent_chunk_id)
                        VALUES
                            (:t, :e, 'T', 'C', 'h',
                             CAST(:emb AS vector), FALSE, NULL)
                        """
                    ),
                    {"t": tenant_id, "e": entry_id, "emb": bad_vector},
                )
    finally:
        await engine.dispose()


async def test_partial_unique_treats_null_section_as_equal(
    fresh_pgvector: PostgresContainer,
) -> None:
    """`uq_kb_chunks_content` uses NULLS NOT DISTINCT so two content
    chunks with the same (tenant, entry, NULL section) collide —
    required for idempotent re-ingestion of unsectioned entries."""
    _alembic(fresh_pgvector, "upgrade", "head")

    from sqlalchemy.exc import IntegrityError

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    zero_vector = "[" + ",".join(["0"] * 1024) + "]"
    try:
        async with engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('t', 't-null') RETURNING id")
                )
            ).scalar_one()
            entry_id = (
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:t, 'k', 'T', 'x') RETURNING id"
                    ),
                    {"t": tenant_id},
                )
            ).scalar_one()
            await conn.execute(
                text(
                    """
                    INSERT INTO kb_chunks
                        (tenant_id, kb_entry_id, section, title, content,
                         content_sha256, embedding, is_synthetic_question,
                         parent_chunk_id)
                    VALUES
                        (:t, :e, NULL, 'T', 'C', 'h1',
                         CAST(:emb AS vector), FALSE, NULL)
                    """
                ),
                {"t": tenant_id, "e": entry_id, "emb": zero_vector},
            )
            with pytest.raises(IntegrityError):
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_chunks
                            (tenant_id, kb_entry_id, section, title, content,
                             content_sha256, embedding, is_synthetic_question,
                             parent_chunk_id)
                        VALUES
                            (:t, :e, NULL, 'T2', 'C2', 'h2',
                             CAST(:emb AS vector), FALSE, NULL)
                        """
                    ),
                    {"t": tenant_id, "e": entry_id, "emb": zero_vector},
                )
    finally:
        await engine.dispose()


async def test_rls_policies_enforce_tenant_isolation_for_non_owner_role(
    fresh_pgvector: PostgresContainer,
) -> None:
    """Issue #41: the RLS migration creates `tenant_isolation_*`
    policies on kb_entries / kb_chunks / retrieval_traces.

    For RLS to actually fire, the connecting role must be NEITHER
    the table owner NOR a superuser (Postgres bypasses RLS for both
    unless FORCE is set, which we deliberately don't use — see
    migration 0006 docstring). This test creates a non-superuser
    role, switches into it via SET ROLE, then confirms:

      (a) SELECT against kb_entries with no GUC returns 0 rows
          (policy denies fail-closed via the COALESCE → zero UUID).
      (b) SELECT with `app.current_tenant_id` set to a tenant
          returns that tenant's rows.
      (c) SELECT with `app.current_tenant_id` set to a DIFFERENT
          tenant returns zero rows from the first tenant.

    Verifies the policy is correctly shaped, not just that the
    migration applied.
    """
    import uuid

    _alembic(fresh_pgvector, "upgrade", "head")

    engine = create_async_engine(_asyncpg_dsn(fresh_pgvector), future=True)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()

    try:
        # Seed two tenants + one kb_entry per tenant under the superuser
        # session (RLS bypassed for owner here — that's the point of NOT
        # FORCEing it).
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO tenants (id, name, slug) VALUES "
                    "(:a, 'TenantA', 'a'), (:b, 'TenantB', 'b')"
                ),
                {"a": tenant_a, "b": tenant_b},
            )
            await conn.execute(
                text(
                    "INSERT INTO kb_entries "
                    "(id, tenant_id, kb_entry_key, title, content_sha256) "
                    "VALUES (:i1, :a, 'main', 'A', 'sha-a'), "
                    "       (:i2, :b, 'main', 'B', 'sha-b')"
                ),
                {"i1": uuid.uuid4(), "i2": uuid.uuid4(), "a": tenant_a, "b": tenant_b},
            )
            # Create a non-superuser role to test the policy under.
            # asyncpg's prepared-statement protocol can't run two SQL
            # commands in one execute(), so issue them separately.
            await conn.execute(text("CREATE ROLE rls_test_user NOLOGIN"))
            await conn.execute(text("GRANT SELECT ON kb_entries TO rls_test_user"))

        # Switch into the non-superuser role and verify policy behavior.
        async with engine.begin() as conn:
            await conn.execute(text("SET ROLE rls_test_user"))

            # (a) GUC unset → 0 rows visible (fail-closed).
            result = await conn.execute(text("SELECT COUNT(*) FROM kb_entries"))
            count = result.scalar_one()
            assert count == 0, (
                f"RLS denied-by-default failed: expected 0 rows, got {count}. "
                f"Policy is likely missing or the role is RLS-bypass."
            )

            # (b) GUC set to tenant_a → tenant_a's row visible.
            await conn.execute(
                text("SELECT set_config('app.current_tenant_id', :tid, true)"),
                {"tid": str(tenant_a)},
            )
            result = await conn.execute(text("SELECT tenant_id FROM kb_entries"))
            visible = {row[0] for row in result.all()}
            assert visible == {tenant_a}, f"RLS scope failed: expected {{tenant_a}}, got {visible}"

            # (c) GUC set to tenant_b → ONLY tenant_b's row visible.
            await conn.execute(
                text("SELECT set_config('app.current_tenant_id', :tid, true)"),
                {"tid": str(tenant_b)},
            )
            result = await conn.execute(text("SELECT tenant_id FROM kb_entries"))
            visible = {row[0] for row in result.all()}
            assert visible == {tenant_b}, (
                f"RLS cross-tenant leak: expected {{tenant_b}}, got {visible}"
            )
    finally:
        await engine.dispose()

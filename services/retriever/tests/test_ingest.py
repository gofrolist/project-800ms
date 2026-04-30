"""Tests for the KB ingestion CLI — content phase.

Hits the session-scoped pgvector testcontainer (real Postgres, real RLS)
but stubs the embedder so we don't pay the BGE-M3 load cost. Each test
provisions its own tenant slug so concurrent / sequential runs don't
collide on the unique constraint.

Coverage targets:

* Idempotency — re-running the same source produces zero writes.
* Add / update / delete plumbing — diff math + side effects.
* Mass-deletion safeguard — fires when delete-rate exceeds threshold.
* Source validation — bad JSON files and missing namespace prefix.
* ``--dry-run`` — no DB writes happen.
* Empty source dir — graceful no-op rather than catastrophic delete.

Synthetic-question generation is exercised by ``test_synthetic_questions.py``;
this file only verifies that the content phase wires up the
``skip_synthetic_questions`` toggle correctly (default = skip in the
public API, until the synth phase lands).
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import stub_encode as _stub_encode


@pytest_asyncio.fixture
async def ingest_env(pgvector_postgres, monkeypatch) -> AsyncIterator[dict[str, Any]]:
    """Set up retriever env + a fresh tenant for an ingestion test.

    Yields a dict with ``tenant_slug`` and ``tenant_id``; the seeded
    tenant survives the test (we don't roll back) so on-disk DB state
    can be inspected directly. Each test gets a unique slug so we can
    reuse the session-scoped container without collisions.
    """
    dsn = pgvector_postgres.get_connection_url().replace("+psycopg2", "+asyncpg")
    monkeypatch.setenv("DB_URL", dsn)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm-test.local")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("REWRITER_MODEL", "test-rewriter-model")
    monkeypatch.setenv("EMBEDDER_DEVICE", "cpu")
    monkeypatch.setenv("RETRIEVER_INTERNAL_TOKEN", "test-internal-token")

    import config
    import db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()

    # Provision a fresh tenant for this test.
    tenant_id = uuid.uuid4()
    tenant_slug = f"ingest-{tenant_id.hex[:8]}"
    seed_engine = create_async_engine(dsn)
    async with seed_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, slug, status) VALUES (:i, :n, :s, 'active')"),
            {"i": tenant_id, "n": "Ingest Test Tenant", "s": tenant_slug},
        )
    await seed_engine.dispose()

    # Drop cached ingest module so it picks up the new env on import.
    sys.modules.pop("ingest", None)

    yield {"tenant_slug": tenant_slug, "tenant_id": tenant_id, "dsn": dsn}

    # Best-effort cleanup so test logs / re-runs don't accumulate dead
    # tenants in a long-lived testcontainer (which is session-scoped).
    cleanup_engine = create_async_engine(dsn)
    async with cleanup_engine.begin() as conn:
        await conn.execute(text("DELETE FROM tenants WHERE id = :i"), {"i": tenant_id})
    await cleanup_engine.dispose()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()


def _write_articles(tmp_path: Path, articles: list[dict[str, Any]]) -> Path:
    """Write a list of canonical articles into a fresh kb dir under tmp."""
    out = tmp_path / "kb"
    out.mkdir(parents=True, exist_ok=True)
    for art in articles:
        # Use the suffix after the namespace prefix as the filename
        # (mirrors fetch_chatwoot_kb.py).
        suffix = art["kb_entry_key"].split(":", 1)[1]
        (out / f"{suffix}.json").write_text(
            json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out


def _article(
    kb_entry_key: str,
    *,
    title: str = "Тестовая статья",
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kb_entry_key": kb_entry_key,
        "title": title,
        "content": content
        if content is not None
        else (
            "## Раздел один\n\nПервое содержательное содержимое статьи.\n\n"
            "## Раздел два\n\nВторое содержательное содержимое статьи.\n"
        ),
        "source_uri": f"https://example.test/{kb_entry_key}",
        "metadata": metadata or {"source": "chatwoot"},
    }


async def _count(dsn: str, sql: str, params: dict[str, Any]) -> int:
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            return int(result.scalar_one())
    finally:
        await engine.dispose()


@pytest.mark.slow
async def test_ingest_adds_new_entries(ingest_env, tmp_path) -> None:
    import ingest

    src = _write_articles(
        tmp_path,
        [
            _article("chatwoot:1"),
            _article("chatwoot:2", title="Вторая"),
        ],
    )

    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"],
        source=src,
        encode_fn=_stub_encode,
    )

    assert summary["entries"]["seen"] == 2
    assert summary["entries"]["added"] == 2
    assert summary["entries"]["updated"] == 0
    assert summary["entries"]["unchanged"] == 0
    assert summary["entries"]["deleted"] == 0
    assert summary["chunks"]["added"] >= 2  # at least one chunk per article
    assert summary["embed_calls"] == summary["chunks"]["added"]
    assert summary["chunks"]["deleted"] == 0
    assert summary["namespace"] == "chatwoot"

    # Verify DB state.
    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 2

    n_chunks = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_chunks WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_chunks == summary["chunks"]["added"]


@pytest.mark.slow
async def test_ingest_is_idempotent_on_unchanged_content(ingest_env, tmp_path) -> None:
    """Second run with identical source must produce zero writes."""
    import ingest

    src = _write_articles(tmp_path, [_article("chatwoot:1"), _article("chatwoot:2")])

    first = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"], source=src, encode_fn=_stub_encode
    )
    assert first["entries"]["added"] == 2

    second = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"], source=src, encode_fn=_stub_encode
    )
    assert second["entries"]["added"] == 0
    assert second["entries"]["updated"] == 0
    assert second["entries"]["unchanged"] == 2
    assert second["entries"]["deleted"] == 0
    assert second["embed_calls"] == 0


@pytest.mark.slow
async def test_ingest_updates_changed_content(ingest_env, tmp_path) -> None:
    import ingest

    src = _write_articles(tmp_path, [_article("chatwoot:1")])
    await ingest.run(tenant_slug=ingest_env["tenant_slug"], source=src, encode_fn=_stub_encode)

    # Rewrite content for the same kb_entry_key.
    src2 = _write_articles(
        tmp_path / "v2",
        [
            _article(
                "chatwoot:1", content="## Новый раздел\n\nСовершенно другое содержимое статьи.\n"
            )
        ],
    )
    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"], source=src2, encode_fn=_stub_encode
    )

    assert summary["entries"]["added"] == 0
    assert summary["entries"]["updated"] == 1
    assert summary["entries"]["unchanged"] == 0

    # Old chunks are gone, new chunks present.
    n_chunks = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_chunks WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_chunks == summary["chunks"]["added"]


@pytest.mark.slow
async def test_ingest_deletes_entries_missing_from_source(ingest_env, tmp_path) -> None:
    import ingest

    # Seed with three; second run drops one.
    full_src = _write_articles(tmp_path, [_article(f"chatwoot:{i}") for i in (1, 2, 3)])
    await ingest.run(tenant_slug=ingest_env["tenant_slug"], source=full_src, encode_fn=_stub_encode)

    smaller_src = _write_articles(tmp_path / "smaller", [_article(f"chatwoot:{i}") for i in (1, 2)])
    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"],
        source=smaller_src,
        encode_fn=_stub_encode,
    )

    # 1 of 3 deleted = 33%, but base is below MASS_DELETION_FLOOR (4),
    # so the safeguard does NOT fire.
    assert summary["entries"]["deleted"] == 1
    assert summary["chunks"]["deleted"] >= 1

    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 2


@pytest.mark.slow
async def test_mass_deletion_safeguard_fires(ingest_env, tmp_path) -> None:
    import ingest

    # Seed with 10 entries.
    full_src = _write_articles(tmp_path, [_article(f"chatwoot:{i}") for i in range(10)])
    await ingest.run(tenant_slug=ingest_env["tenant_slug"], source=full_src, encode_fn=_stub_encode)

    # New source has only 3 — would delete 7 of 10 = 70% > 25%. Safeguard
    # fires (SystemExit with EXIT_PARTIAL_SUCCESS).
    tiny_src = _write_articles(tmp_path / "tiny", [_article(f"chatwoot:{i}") for i in range(3)])
    with pytest.raises(SystemExit) as exc:
        await ingest.run(
            tenant_slug=ingest_env["tenant_slug"],
            source=tiny_src,
            encode_fn=_stub_encode,
        )
    assert exc.value.code == ingest.EXIT_PARTIAL_SUCCESS

    # DB unchanged — safeguard fires before any writes happen.
    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 10


@pytest.mark.slow
async def test_mass_deletion_override_works(ingest_env, tmp_path) -> None:
    import ingest

    full_src = _write_articles(tmp_path, [_article(f"chatwoot:{i}") for i in range(10)])
    await ingest.run(tenant_slug=ingest_env["tenant_slug"], source=full_src, encode_fn=_stub_encode)

    tiny_src = _write_articles(tmp_path / "tiny", [_article(f"chatwoot:{i}") for i in range(3)])
    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"],
        source=tiny_src,
        encode_fn=_stub_encode,
        allow_mass_deletion=True,
    )
    assert summary["entries"]["deleted"] == 7

    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 3


@pytest.mark.slow
async def test_dry_run_writes_nothing(ingest_env, tmp_path) -> None:
    import ingest

    src = _write_articles(tmp_path, [_article("chatwoot:1"), _article("chatwoot:2")])
    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"],
        source=src,
        dry_run=True,
        encode_fn=_stub_encode,
    )

    assert summary["dry_run"] is True
    assert summary["entries"]["added"] == 2
    assert summary["embed_calls"] == 0
    assert summary["chunks"]["added"] == 0  # nothing actually written

    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 0


@pytest.mark.slow
async def test_empty_source_dir_is_noop_not_catastrophe(ingest_env, tmp_path) -> None:
    """Empty fetch dir must not cascade-delete the existing KB.

    This is the "Chatwoot returned 0 articles because auth flipped"
    scenario. The mass-deletion safeguard would catch it indirectly,
    but we treat 'no source entries at all' as a hard short-circuit so
    operators see a clear log line instead of a vague safeguard exit.
    """
    import ingest

    # Pre-seed the tenant.
    src = _write_articles(tmp_path, [_article("chatwoot:1"), _article("chatwoot:2")])
    await ingest.run(tenant_slug=ingest_env["tenant_slug"], source=src, encode_fn=_stub_encode)

    empty = tmp_path / "empty"
    empty.mkdir()
    summary = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"], source=empty, encode_fn=_stub_encode
    )

    assert summary["entries"]["seen"] == 0
    assert summary["entries"]["deleted"] == 0  # no diff was performed

    # Existing entries still there.
    n_entries = await _count(
        ingest_env["dsn"],
        "SELECT count(*) FROM kb_entries WHERE tenant_id = :t",
        {"t": ingest_env["tenant_id"]},
    )
    assert n_entries == 2


@pytest.mark.slow
async def test_unknown_tenant_exits_with_bad_args(ingest_env, tmp_path) -> None:
    import ingest

    src = _write_articles(tmp_path, [_article("chatwoot:1")])
    with pytest.raises(SystemExit) as exc:
        await ingest.run(
            tenant_slug="no-such-tenant-9999",
            source=src,
            encode_fn=_stub_encode,
        )
    assert exc.value.code == ingest.EXIT_BAD_ARGS


@pytest.mark.slow
async def test_full_mode_re_embeds_unchanged_entries(ingest_env, tmp_path) -> None:
    """``--mode full`` forces re-embed even when content_sha256 matches."""
    import ingest

    src = _write_articles(tmp_path, [_article("chatwoot:1")])
    first = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"], source=src, encode_fn=_stub_encode
    )
    assert first["entries"]["added"] == 1

    second = await ingest.run(
        tenant_slug=ingest_env["tenant_slug"],
        source=src,
        mode="full",
        encode_fn=_stub_encode,
    )
    # Existing entry treated as updated in full mode.
    assert second["entries"]["updated"] == 1
    assert second["entries"]["unchanged"] == 0
    assert second["embed_calls"] >= 1


def test_mixed_namespaces_in_source_aborts(tmp_path) -> None:
    """Two namespaces in one source dir triggers EXIT_PARSE_ERROR."""
    import ingest

    out = tmp_path / "kb"
    out.mkdir()
    (out / "a.json").write_text(
        json.dumps(_article("chatwoot:1"), ensure_ascii=False), encoding="utf-8"
    )
    (out / "b.json").write_text(
        json.dumps(_article("wiki:2"), ensure_ascii=False), encoding="utf-8"
    )

    import asyncio as _asyncio

    with pytest.raises(SystemExit) as exc:
        _asyncio.run(
            ingest.run(
                tenant_slug="anything",  # never reached
                source=out,
                encode_fn=_stub_encode,
            )
        )
    assert exc.value.code == ingest.EXIT_PARSE_ERROR


def test_load_source_dir_reports_parse_errors(tmp_path) -> None:
    """Bad JSON / missing fields land in the errors list, not raised."""
    import ingest

    out = tmp_path / "kb"
    out.mkdir()
    (out / "good.json").write_text(
        json.dumps(_article("chatwoot:1"), ensure_ascii=False), encoding="utf-8"
    )
    (out / "bad.json").write_text("{this is not json", encoding="utf-8")
    (out / "missing-namespace.json").write_text(
        json.dumps({"kb_entry_key": "nosep", "title": "x", "content": "y"}),
        encoding="utf-8",
    )

    entries, errors = ingest._load_source_dir(out)
    assert len(entries) == 1
    assert entries[0].kb_entry_key == "chatwoot:1"
    assert len(errors) == 2  # bad.json + missing-namespace.json


def test_diff_categorises_correctly() -> None:
    import ingest

    db = {
        "chatwoot:keep": {"content_sha256": "AAA"},
        "chatwoot:change": {"content_sha256": "OLD"},
        "chatwoot:gone": {"content_sha256": "ZZZ"},
    }
    src = [
        ingest.SourceEntry(
            kb_entry_key="chatwoot:keep",
            title="t",
            content="c",
            content_sha256="AAA",
            source_uri=None,
            metadata={},
            namespace="chatwoot",
        ),
        ingest.SourceEntry(
            kb_entry_key="chatwoot:change",
            title="t",
            content="c2",
            content_sha256="NEW",
            source_uri=None,
            metadata={},
            namespace="chatwoot",
        ),
        ingest.SourceEntry(
            kb_entry_key="chatwoot:new",
            title="t",
            content="c3",
            content_sha256="BBB",
            source_uri=None,
            metadata={},
            namespace="chatwoot",
        ),
    ]

    plan = ingest._diff(src, db, "chatwoot")
    assert [e.kb_entry_key for e in plan.add] == ["chatwoot:new"]
    assert [e.kb_entry_key for e in plan.update] == ["chatwoot:change"]
    assert [e.kb_entry_key for e in plan.unchanged] == ["chatwoot:keep"]
    assert plan.delete == ["chatwoot:gone"]

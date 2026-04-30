"""Tests for the synthetic-question generation phase.

Stubs the LLM via ``respx`` so we don't hit Groq during CI. Hits the
session-scoped pgvector testcontainer so the SQL (parent_chunk_id FK,
is_synthetic_question filter, NOT EXISTS naked-chunk query) actually
runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import stub_encode as _stub_encode


@pytest_asyncio.fixture
async def synth_env(pgvector_postgres, monkeypatch) -> AsyncIterator[dict[str, Any]]:
    """Spin up a fresh tenant + ingest some content so synth has work."""
    dsn = pgvector_postgres.get_connection_url().replace("+psycopg2", "+asyncpg")
    monkeypatch.setenv("DB_URL", dsn)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm-test.local")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("REWRITER_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDER_DEVICE", "cpu")
    monkeypatch.setenv("RETRIEVER_INTERNAL_TOKEN", "test-internal-token")

    import config
    import db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()

    tenant_id = uuid.uuid4()
    tenant_slug = f"synth-{tenant_id.hex[:8]}"
    seed_engine = create_async_engine(dsn)
    async with seed_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, slug, status) VALUES (:i, :n, :s, 'active')"),
            {"i": tenant_id, "n": "Synth Test", "s": tenant_slug},
        )
    await seed_engine.dispose()

    sys.modules.pop("ingest", None)
    sys.modules.pop("synthetic_questions", None)

    yield {"tenant_slug": tenant_slug, "tenant_id": tenant_id, "dsn": dsn}

    cleanup_engine = create_async_engine(dsn)
    async with cleanup_engine.begin() as conn:
        await conn.execute(text("DELETE FROM tenants WHERE id = :i"), {"i": tenant_id})
    await cleanup_engine.dispose()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()


def _ingest_simple(tmp_path: Path) -> Path:
    """Write one short article to ingest as content so synth has chunks."""
    out = tmp_path / "kb"
    out.mkdir(parents=True, exist_ok=True)
    (out / "1.json").write_text(
        json.dumps(
            {
                "kb_entry_key": "chatwoot:1",
                "title": "Цены на права",
                "content": (
                    "## Получение прав\n\n"
                    "Чтобы получить водительские права, посетите автошколу "
                    "и сдайте экзамены. Стоимость обучения — 5 000 долларов.\n"
                ),
                "source_uri": "https://example.test/1",
                "metadata": {"source": "chatwoot"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


@pytest.mark.slow
async def test_synth_inserts_questions_with_parent_chunk_id(synth_env, tmp_path) -> None:
    """Happy path: LLM returns 4 questions; each gets inserted as
    is_synthetic_question=True with parent_chunk_id linking to the
    content chunk."""
    import ingest
    import synthetic_questions

    src = _ingest_simple(tmp_path)
    await ingest.run(
        tenant_slug=synth_env["tenant_slug"],
        source=src,
        encode_fn=_stub_encode,
        skip_synthetic_questions=True,
    )

    chat_url = "http://llm-test.local/chat/completions"
    questions = [
        "Сколько стоит обучение в автошколе?",
        "Где получить водительские права?",
        "Какие экзамены сдавать на права?",
        "Цена за получение прав?",
    ]
    llm_body = json.dumps({"questions": questions}, ensure_ascii=False)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(chat_url).respond(
            200,
            json={"choices": [{"message": {"content": llm_body}}]},
        )

        async with httpx.AsyncClient() as client:
            summary = await synthetic_questions.run(
                tenant_slug=synth_env["tenant_slug"],
                namespace="chatwoot",
                encode_fn=_stub_encode,
                http_client=client,
            )

    assert summary["added"] == 4
    assert summary["chunks_touched"] == 1
    assert summary["rewriter_calls"] == 1

    # DB invariants.
    engine = create_async_engine(synth_env["dsn"])
    try:
        async with engine.connect() as conn:
            n_synth = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM kb_chunks "
                        "WHERE tenant_id = :t AND is_synthetic_question = TRUE"
                    ),
                    {"t": synth_env["tenant_id"]},
                )
            ).scalar_one()
            n_orphan = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM kb_chunks "
                        "WHERE tenant_id = :t AND is_synthetic_question = TRUE "
                        "  AND parent_chunk_id IS NULL"
                    ),
                    {"t": synth_env["tenant_id"]},
                )
            ).scalar_one()
            n_self_parent = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM kb_chunks "
                        "WHERE tenant_id = :t AND is_synthetic_question = TRUE "
                        "  AND parent_chunk_id = id"
                    ),
                    {"t": synth_env["tenant_id"]},
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert n_synth == 4
    assert n_orphan == 0  # invariant: every synth row has a parent
    assert n_self_parent == 0  # parent must point at a content chunk


@pytest.mark.slow
async def test_synth_is_idempotent_on_rerun(synth_env, tmp_path) -> None:
    """Second run with no new naked chunks must not call the LLM again."""
    import ingest
    import synthetic_questions

    src = _ingest_simple(tmp_path)
    await ingest.run(
        tenant_slug=synth_env["tenant_slug"],
        source=src,
        encode_fn=_stub_encode,
        skip_synthetic_questions=True,
    )

    chat_url = "http://llm-test.local/chat/completions"
    llm_body = json.dumps({"questions": ["q1?", "q2?", "q3?", "q4?"]}, ensure_ascii=False)

    with respx.mock(assert_all_called=False) as mock:
        chat_route = mock.post(chat_url).respond(
            200, json={"choices": [{"message": {"content": llm_body}}]}
        )

        async with httpx.AsyncClient() as client:
            first = await synthetic_questions.run(
                tenant_slug=synth_env["tenant_slug"],
                namespace="chatwoot",
                encode_fn=_stub_encode,
                http_client=client,
            )
            assert first["added"] == 4
            calls_after_first = chat_route.call_count

            second = await synthetic_questions.run(
                tenant_slug=synth_env["tenant_slug"],
                namespace="chatwoot",
                encode_fn=_stub_encode,
                http_client=client,
            )

    # Second run sees no naked chunks → zero LLM calls + zero new rows.
    assert second["added"] == 0
    assert second["chunks_touched"] == 0
    assert chat_route.call_count == calls_after_first


@pytest.mark.slow
async def test_synth_aborts_after_consecutive_429s(synth_env, tmp_path) -> None:
    """Persistent rate-limit responses → phase aborts gracefully, leaving
    content chunks intact and naked for the next run."""
    import ingest
    import synthetic_questions

    # Ingest 8 short articles → 8 content chunks, all naked.
    out = tmp_path / "kb"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        (out / f"{i}.json").write_text(
            json.dumps(
                {
                    "kb_entry_key": f"chatwoot:{i}",
                    "title": f"Статья {i}",
                    "content": f"## Раздел\n\nСодержательное содержимое статьи номер {i}.\n",
                    "source_uri": None,
                    "metadata": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    await ingest.run(
        tenant_slug=synth_env["tenant_slug"],
        source=out,
        encode_fn=_stub_encode,
        skip_synthetic_questions=True,
    )

    chat_url = "http://llm-test.local/chat/completions"
    with respx.mock(assert_all_called=False) as mock:
        mock.post(chat_url).respond(429, json={"error": "rate_limited"})

        async with httpx.AsyncClient() as client:
            # Patch sleep so the test doesn't actually wait between
            # backoffs (3 attempts × 0.5s + 1s + 2s = ~3.5s/chunk = 28s
            # total without this patch).
            import asyncio as _asyncio

            real_sleep = _asyncio.sleep

            async def fast_sleep(_seconds: float) -> None:
                await real_sleep(0)

            mp = pytest.MonkeyPatch()
            try:
                mp.setattr(synthetic_questions.asyncio, "sleep", fast_sleep)
                summary = await synthetic_questions.run(
                    tenant_slug=synth_env["tenant_slug"],
                    namespace="chatwoot",
                    encode_fn=_stub_encode,
                    http_client=client,
                )
            finally:
                mp.undo()

    # Phase aborted after 5 consecutive failures (5 chunks touched, 0
    # added). Remaining chunks stay naked for the next run.
    assert summary["added"] == 0
    assert summary["chunks_touched"] >= 5
    assert summary["chunks_touched"] <= 8


@pytest.mark.slow
async def test_synth_skips_chunks_with_existing_children(synth_env, tmp_path) -> None:
    """A pre-existing synthetic-question row protects its parent from
    re-synthesis even on the first synth run."""
    import ingest
    import synthetic_questions

    src = _ingest_simple(tmp_path)
    await ingest.run(
        tenant_slug=synth_env["tenant_slug"],
        source=src,
        encode_fn=_stub_encode,
        skip_synthetic_questions=True,
    )

    # Manually insert a synth row pointing at the existing content chunk
    # so the NOT EXISTS filter excludes it.
    engine = create_async_engine(synth_env["dsn"])
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_tenant_id', :t, true)"),
            {"t": str(synth_env["tenant_id"])},
        )
        chunk_row = (
            (
                await conn.execute(
                    text(
                        "SELECT id, kb_entry_id, title, section "
                        "FROM kb_chunks WHERE tenant_id = :t "
                        "  AND is_synthetic_question = FALSE LIMIT 1"
                    ),
                    {"t": synth_env["tenant_id"]},
                )
            )
            .mappings()
            .first()
        )
        assert chunk_row is not None
        sha = hashlib.sha256(b"manual-q").hexdigest()
        emb = "[" + ",".join("0.1" for _ in range(1024)) + "]"
        await conn.execute(
            text(
                "INSERT INTO kb_chunks "
                "  (tenant_id, kb_entry_id, section, title, content, "
                "   content_sha256, embedding, is_synthetic_question, "
                "   parent_chunk_id) "
                "VALUES (:t, :e, :s, :ti, 'manual?', :sha, "
                "        CAST(:emb AS vector), TRUE, :pid)"
            ),
            {
                "t": synth_env["tenant_id"],
                "e": chunk_row["kb_entry_id"],
                "s": chunk_row["section"],
                "ti": chunk_row["title"],
                "sha": sha,
                "emb": emb,
                "pid": chunk_row["id"],
            },
        )
    await engine.dispose()

    chat_url = "http://llm-test.local/chat/completions"
    with respx.mock(assert_all_called=False) as mock:
        chat_route = mock.post(chat_url).respond(
            200,
            json={"choices": [{"message": {"content": '{"questions": ["?"]}'}}]},
        )
        async with httpx.AsyncClient() as client:
            summary = await synthetic_questions.run(
                tenant_slug=synth_env["tenant_slug"],
                namespace="chatwoot",
                encode_fn=_stub_encode,
                http_client=client,
            )

    assert summary["chunks_touched"] == 0
    assert chat_route.call_count == 0


@pytest.mark.slow
async def test_synth_unknown_tenant_returns_empty_summary(synth_env) -> None:
    """Unknown slug = no-op, never raises."""
    import synthetic_questions

    async with httpx.AsyncClient() as client:
        summary = await synthetic_questions.run(
            tenant_slug="nope-9999",
            namespace="chatwoot",
            encode_fn=_stub_encode,
            http_client=client,
        )
    assert summary["added"] == 0
    assert summary["chunks_touched"] == 0

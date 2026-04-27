"""SC-001 eval harness — top-3 recall on curated Russian in-scope queries.

Usage:
    cd services/retriever
    uv run python -m evals.run_eval                # defaults threshold=0.80
    uv run python -m evals.run_eval --threshold 0.9

Exit code 0 if recall ≥ threshold (SC-001 = 0.80), 1 otherwise. The
failing queries are printed so a regression is immediately debuggable.

Self-contained: spins up a pgvector/pgvector:pg18 testcontainer, runs
the apps/api migrations, seeds the tenant + KB from `sample_kb.yaml`,
mocks the rewriter LLM to PASS THROUGH the transcript verbatim (so the
eval isolates retrieval quality from rewriter quality), and exercises
the real `/retrieve` endpoint via an ASGI client.

Takes ~1–2 min end-to-end once the BGE-M3 weights are cached in
`hf_cache_retriever`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import respx
import yaml
from httpx import ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "apps" / "api"
_EVALS_DIR = Path(__file__).parent

_LLM_BASE = "http://eval-llm.local"
_LLM_URL = f"{_LLM_BASE}/chat/completions"


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10g}" for x in v) + "]"


async def _seed_kb(
    dsn: str,
    tenant_id: uuid.UUID,
    session_id: uuid.UUID,
    api_key_id: uuid.UUID,
    kb_data: dict,
) -> dict[str, uuid.UUID]:
    """Ingest `sample_kb.yaml` into the container and return
    `entry_key → kb_entries.id` so the eval can match chunks to
    expected-entry tags."""
    from embedder import encode, preload

    preload()

    # Embed all chunks OUTSIDE the transaction so the COMMIT is brief.
    precomputed = []
    for entry in kb_data["entries"]:
        for chunk in entry["chunks"]:
            emb = await encode(chunk["content"])
            precomputed.append(
                {
                    "entry_key": entry["key"],
                    "entry_title": entry["title"],
                    "section": chunk["section"],
                    "content": chunk["content"],
                    "embedding": _vector_literal(emb),
                }
            )

    engine = create_async_engine(dsn)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO tenants (id, name, slug) VALUES (:i, 'Eval', :s)"),
                {"i": tenant_id, "s": f"eval-{tenant_id.hex[:8]}"},
            )
            await conn.execute(
                text(
                    "INSERT INTO api_keys (id, tenant_id, key_hash, key_prefix) "
                    "VALUES (:i, :t, decode(:h, 'hex'), :p)"
                ),
                {
                    "i": api_key_id,
                    "t": tenant_id,
                    "h": secrets.token_hex(32),
                    "p": "evalkey",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO sessions (id, tenant_id, api_key_id, room, identity) "
                    "VALUES (:si, :t, :ai, :r, 'eval-user')"
                ),
                {
                    "si": session_id,
                    "t": tenant_id,
                    "ai": api_key_id,
                    "r": f"room-{session_id.hex[:8]}",
                },
            )

            entries_by_key: dict[str, uuid.UUID] = {}
            for entry in kb_data["entries"]:
                entry_id = uuid.uuid4()
                entries_by_key[entry["key"]] = entry_id
                await conn.execute(
                    text(
                        "INSERT INTO kb_entries "
                        "(id, tenant_id, kb_entry_key, title, content_sha256) "
                        "VALUES (:i, :t, :k, :title, :sha)"
                    ),
                    {
                        "i": entry_id,
                        "t": tenant_id,
                        "k": entry["key"],
                        "title": entry["title"],
                        "sha": f"sha-{entry['key']}",
                    },
                )

            for row in precomputed:
                entry_id = entries_by_key[row["entry_key"]]
                await conn.execute(
                    text(
                        "INSERT INTO kb_chunks "
                        "(tenant_id, kb_entry_id, section, title, content, "
                        " content_sha256, embedding) "
                        "VALUES (:t, :e, :sec, :title, :content, :sha, "
                        "        CAST(:emb AS vector))"
                    ),
                    {
                        "t": tenant_id,
                        "e": entry_id,
                        "sec": row["section"],
                        "title": f"{row['entry_title']} — {row['section']}",
                        "content": row["content"],
                        "sha": f"sha-{row['entry_key']}-{row['section']}",
                        "emb": row["embedding"],
                    },
                )
    finally:
        await engine.dispose()

    return entries_by_key


def _install_passthrough_rewriter_mock() -> None:
    """Respond to the rewriter LLM as if it returned the transcript
    unchanged, always in-scope. Isolates retrieval quality from the
    rewriter's behavior so an eval failure is unambiguously a retrieval
    problem."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query": last, "in_scope": True}),
                        }
                    }
                ]
            },
        )

    respx.post(_LLM_URL).mock(side_effect=handler)


async def _run_queries(
    app,
    tenant_id: uuid.UUID,
    session_id: uuid.UUID,
    questions: list[dict],
) -> list[dict]:
    """POST /retrieve for each question; return a per-row pass/fail record.

    Match logic: a query is judged correct when top-3 contains a chunk
    whose title suffix matches the expected section. Title format is
    `{entry_title} — {section}` per `_seed_kb`, so the suffix check
    implicitly ties (entry_key, section) back to a specific chunk.
    """
    results: list[dict] = []
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://eval.retriever.test",
        timeout=30.0,
    ) as client:
        for q in questions:
            resp = await client.post(
                "/retrieve",
                json={
                    "tenant_id": str(tenant_id),
                    "session_id": str(session_id),
                    "turn_id": q["id"],
                    "transcript": q["query"],
                    "top_k": 3,
                },
            )
            if resp.status_code != 200:
                results.append(
                    {
                        "id": q["id"],
                        "ok": False,
                        "reason": f"HTTP {resp.status_code}",
                        "query": q["query"],
                        "expected": f"{q['expected_kb_entry_key']}/{q['expected_section']}",
                    }
                )
                continue

            body = resp.json()
            top_3 = body["chunks"][:3]
            expected_section = q["expected_section"]
            title_suffix = f"— {expected_section}"
            found = any(title_suffix in c["title"] for c in top_3)

            results.append(
                {
                    "id": q["id"],
                    "ok": found,
                    "query": q["query"],
                    "expected": f"{q['expected_kb_entry_key']}/{q['expected_section']}",
                    "top_3_titles": [c["title"] for c in top_3],
                }
            )

    return results


async def _main(args: argparse.Namespace) -> int:
    with open(args.kb) as f:
        kb_data = yaml.safe_load(f)
    with open(args.questions) as f:
        questions = yaml.safe_load(f)["questions"]

    print(
        f"[eval] loading {len(kb_data['entries'])} KB entries "
        f"({sum(len(e['chunks']) for e in kb_data['entries'])} chunks), "
        f"{len(questions)} questions",
        flush=True,
    )

    # Prime required settings BEFORE anything imports the retriever
    # modules — the embedder singleton reads settings on first load,
    # and `_seed_kb` triggers that load to embed KB chunks.
    os.environ.setdefault("DB_URL", "postgresql+asyncpg://placeholder/placeholder")
    os.environ["LLM_BASE_URL"] = _LLM_BASE
    os.environ["LLM_API_KEY"] = "eval-key"
    os.environ["REWRITER_MODEL"] = "eval-model"
    os.environ.setdefault("EMBEDDER_DEVICE", "cpu")

    print("[eval] starting pgvector container...", flush=True)
    container = PostgresContainer(
        image="pgvector/pgvector:pg18",
        username="eval",
        password="eval",
        dbname="eval",
    )
    container.start()
    try:
        dsn = container.get_connection_url().replace("+psycopg2", "+asyncpg")

        print("[eval] running migrations...", flush=True)
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env.update(
            {
                "DATABASE_URL": dsn,
                "LIVEKIT_API_KEY": env.get("LIVEKIT_API_KEY", "eval-livekit"),
                "LIVEKIT_API_SECRET": env.get("LIVEKIT_API_SECRET", "x" * 32),
            }
        )
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=_API_DIR,
            env=env,
            check=True,
        )

        tenant_id = uuid.uuid4()
        session_id = uuid.uuid4()
        api_key_id = uuid.uuid4()
        print("[eval] seeding KB (embedding chunks)...", flush=True)
        await _seed_kb(dsn, tenant_id, session_id, api_key_id, kb_data)

        # Now that the container is up, point the in-process app at
        # its real DSN. Cached factories must be invalidated so the new
        # env takes effect.
        os.environ["DB_URL"] = dsn

        import config
        import db

        config.get_settings.cache_clear()
        db.get_engine.cache_clear()
        db._session_factory.cache_clear()
        sys.modules.pop("main", None)
        sys.modules.pop("retrieve", None)
        from main import create_app

        app = create_app()

        print(f"[eval] running {len(questions)} queries via /retrieve...", flush=True)
        with respx.mock:
            _install_passthrough_rewriter_mock()
            results = await _run_queries(app, tenant_id, session_id, questions)

        correct = sum(1 for r in results if r["ok"])
        total = len(results)
        recall = correct / total if total else 0.0

        print(f"\n[eval] top-3 recall: {correct}/{total} = {recall:.1%}")
        print(f"[eval] threshold:    {args.threshold:.0%} (SC-001)")

        failures = [r for r in results if not r["ok"]]
        if failures:
            print(f"\n[eval] {len(failures)} failures:")
            for r in failures:
                print(f"  {r['id']:>5} expected={r['expected']}")
                print(f"        query={r['query']!r}")
                if r.get("top_3_titles"):
                    for t in r["top_3_titles"]:
                        print(f"        got: {t}")
                elif r.get("reason"):
                    print(f"        reason: {r['reason']}")

        return 0 if recall >= args.threshold else 1

    finally:
        container.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb",
        default=str(_EVALS_DIR / "sample_kb.yaml"),
        help="Path to sample KB YAML (ingested fresh per run).",
    )
    parser.add_argument(
        "--questions",
        default=str(_EVALS_DIR / "inscope_ru.yaml"),
        help="Path to the query fixture YAML.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Recall threshold below which the eval exits non-zero (SC-001 = 0.80).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()

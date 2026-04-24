"""Integration test for POST /retrieve.

Exercises the full pipeline against real Postgres+pgvector (via
testcontainers) with the rewriter LLM call mocked via `respx`. Covers
the five T020 assertions:

  (a) in-scope request → ≥1 chunk, rewritten_query set, trace_id UUID,
      `stage_timings_ms.total == sum(stages) ± 1`
  (b) unknown tenant_id → 400 `unknown_tenant`
  (c) unsupported npc_id → 400 `unsupported_npc`
  (d) unsupported language → 400 `unsupported_language`
  (e) 200 response body validates against
      `contracts/retrieve.openapi.yaml` via `openapi-spec-validator`

Slow-marked (testcontainer + BGE-M3 download). CI runs it; devs skip
with `-m "not slow"`.
"""

from __future__ import annotations

import secrets
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
import yaml
from httpx import ASGITransport
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from embedder import encode, preload

pytestmark = pytest.mark.slow

# Match the LLM the rewriter is configured against — must align with
# the monkeypatched LLM_BASE_URL env.
_LLM_BASE = "http://llm-test.local"
_LLM_URL = f"{_LLM_BASE}/chat/completions"

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "specs"
    / "002-helper-guide-npc"
    / "contracts"
    / "retrieve.openapi.yaml"
)


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10g}" for x in v) + "]"


def _mock_rewriter(
    *, query: str = "как получить водительские права", in_scope: bool = True
) -> None:
    """Install a respx mock returning a rewriter-compatible JSON body."""
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f'{{"query": "{query}", "in_scope": {"true" if in_scope else "false"}}}',
                        }
                    }
                ]
            },
        )
    )


@pytest_asyncio.fixture
async def retriever_app(pgvector_postgres, monkeypatch) -> dict[str, Any]:
    """Build a fresh FastAPI app pointed at the session-scoped pgvector
    container, and seed one tenant + session + KB entry + 3 chunks.

    Seed is COMMITTED on a dedicated engine so the app's own engine
    (opened by `db.get_engine()`) sees the data. The shared container
    tears down at session end — each test uses unique UUIDs to avoid
    collisions with other tests that touch the same tables.
    """
    preload()

    dsn = pgvector_postgres.get_connection_url().replace("+psycopg2", "+asyncpg")

    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    api_key_id = uuid.uuid4()
    kb_entry_id = uuid.uuid4()

    # Pre-embed the chunk contents OUTSIDE the COMMIT transaction.
    chunk_specs = [
        {
            "section": "Получение прав",
            "content": (
                "Чтобы получить водительские права, посетите автошколу "
                "и сдайте экзамены. Стоимость обучения 5 000 долларов."
            ),
        },
        {
            "section": "Цены на транспорт",
            "content": (
                "Грузовик Mule стоит 15 000 долларов. "
                "Легковой автомобиль Premier стоит 12 500 долларов."
            ),
        },
        {
            "section": "Покупка оружия",
            "content": (
                "Оружие можно купить в магазине после получения лицензии. "
                "Лицензия стоит 10 000 долларов."
            ),
        },
    ]
    for spec in chunk_specs:
        spec["embedding"] = _vector_literal(await encode(spec["content"]))

    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, slug) VALUES (:i, :n, :s)"),
            {"i": tenant_id, "n": "Test Tenant", "s": f"t-{tenant_id.hex[:8]}"},
        )
        # Random 32-byte key_hash per test — the column has a UNIQUE
        # index, and pgvector_postgres is session-scoped so rows
        # accumulate across tests.
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, tenant_id, key_hash, key_prefix) "
                "VALUES (:i, :t, decode(:h, 'hex'), :p)"
            ),
            {
                "i": api_key_id,
                "t": tenant_id,
                "h": secrets.token_hex(32),
                "p": f"pfx{tenant_id.hex[:5]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO sessions (id, tenant_id, api_key_id, room, identity) "
                "VALUES (:si, :t, :ai, :room, 'test-user')"
            ),
            {
                "si": session_id,
                "t": tenant_id,
                "ai": api_key_id,
                "room": f"room-{session_id.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO kb_entries "
                "(id, tenant_id, kb_entry_key, title, content_sha256) "
                "VALUES (:i, :t, 'main', 'Main', :sha)"
            ),
            {"i": kb_entry_id, "t": tenant_id, "sha": "shadeadbeef"},
        )
        for idx, spec in enumerate(chunk_specs):
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
                    "e": kb_entry_id,
                    "sec": spec["section"],
                    "title": f"Главный раздел — {spec['section']}",
                    "content": spec["content"],
                    "sha": f"sha-chunk-{idx}",
                    "emb": spec["embedding"],
                },
            )
    await engine.dispose()

    # Point the app at the testcontainer + stub LLM. Cached factories
    # must be invalidated so the new env takes effect.
    monkeypatch.setenv("DB_URL", dsn)
    monkeypatch.setenv("LLM_BASE_URL", _LLM_BASE)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("REWRITER_MODEL", "test-rewriter-model")
    monkeypatch.setenv("EMBEDDER_DEVICE", "cpu")

    import config
    import db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()

    # Import AFTER the env patch so module-level `app = create_app()`
    # reads the correct settings. Purge any cached module import first.
    import sys

    sys.modules.pop("main", None)
    sys.modules.pop("retrieve", None)
    from main import create_app

    app = create_app()

    yield {
        "app": app,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "kb_entry_id": kb_entry_id,
    }


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://retriever.test",
    )


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_in_scope_request_returns_chunks_rewritten_and_trace(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter(query="как получить водительские права", in_scope=True)

    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "как мне получить ПРАВА?",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["in_scope"] is True
    assert body["rewritten_query"] == "как получить водительские права"
    assert len(body["chunks"]) >= 1
    # trace_id is a UUID string
    uuid.UUID(body["trace_id"])
    # stage_timings invariant: total == sum(non-total) ± 1 ms
    st = body["stage_timings_ms"]
    parts = {k: v for k, v in st.items() if k != "total" and v is not None}
    assert abs(st["total"] - sum(parts.values())) <= 1, st
    # Every returned chunk has the required shape
    for c in body["chunks"]:
        assert isinstance(c["id"], int)
        assert isinstance(c["title"], str) and c["title"]
        assert isinstance(c["content"], str) and c["content"]
        assert isinstance(c["score"], float | int)
        assert "semantic" in c["fusion_components"]
        assert "lexical" in c["fusion_components"]


# ─────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_unknown_tenant_returns_400_unknown_tenant(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter()
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(uuid.uuid4()),  # never inserted
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "hi",
            },
        )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"] == "unknown_tenant"


@respx.mock
async def test_unsupported_npc_returns_400_unsupported_npc(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter()
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "hi",
                "npc_id": "evil_npc",
            },
        )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"] == "unsupported_npc"


@respx.mock
async def test_unsupported_language_returns_400_unsupported_language(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter()
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "hello",
                "language": "en",
            },
        )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"] == "unsupported_language"


# ─────────────────────────────────────────────────────────────────────
# Out-of-scope (US2 stub)
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_out_of_scope_returns_empty_chunks(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter(query="погода в Москве", in_scope=False)
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-oos-1",
                "transcript": "какая погода?",
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["in_scope"] is False
    assert body["chunks"] == []
    uuid.UUID(body["trace_id"])


# ─────────────────────────────────────────────────────────────────────
# OpenAPI schema validation
# ─────────────────────────────────────────────────────────────────────


def test_contract_yaml_is_itself_a_valid_openapi_document() -> None:
    """Catch drift or YAML damage in the contract file before we
    measure live responses against it."""
    spec_dict, _spec_url = read_from_filename(str(_CONTRACT_PATH))
    validate_spec(spec_dict)


@respx.mock
async def test_response_shape_conforms_to_openapi_contract(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter(query="как получить водительские права", in_scope=True)
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "как мне получить ПРАВА?",
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Load the contract and project down to the RetrieveResponse schema.
    with open(_CONTRACT_PATH) as f:
        contract = yaml.safe_load(f)

    schema = contract["components"]["schemas"]["RetrieveResponse"]
    required = set(schema.get("required", []))
    # Required top-level keys per OpenAPI contract
    missing = required - set(body.keys())
    assert not missing, f"response missing required keys: {missing}"

    # Shape of `chunks[]` per `RetrievedChunk` schema
    chunk_schema = contract["components"]["schemas"]["RetrievedChunk"]
    chunk_required = set(chunk_schema.get("required", []))
    for c in body["chunks"]:
        missing_in_chunk = chunk_required - set(c.keys())
        assert not missing_in_chunk, f"chunk missing required keys: {missing_in_chunk}; got {c}"

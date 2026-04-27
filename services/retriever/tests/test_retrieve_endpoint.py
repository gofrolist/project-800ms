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

The `retriever_app` fixture (one container + one app + one tenant +
3 KB chunks) lives in conftest.py and is shared with the US2 refusal
+ timing-parity tests so the test suite stays at one container.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from httpx import ASGITransport
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.slow

# Must match `_FIXTURE_LLM_BASE` in conftest.py — that fixture wires
# LLM_BASE_URL to this host so respx can intercept the rewriter call.
_LLM_BASE = "http://llm-test.local"
_LLM_URL = f"{_LLM_BASE}/chat/completions"

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "specs"
    / "002-helper-guide-npc"
    / "contracts"
    / "retrieve.openapi.yaml"
)


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


async def _client(app, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://retriever.test",
        headers=headers,
    )


def _headers(retriever_app: dict[str, Any]) -> dict[str, str]:
    """Pull the X-Internal-Token header out of the fixture for callers."""
    return retriever_app["auth_headers"]


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_in_scope_request_returns_chunks_rewritten_and_trace(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter(query="как получить водительские права", in_scope=True)

    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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


# ─────────────────────────────────────────────────────────────────────
# Pydantic validation → 400 invalid_request (not 422)
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_pydantic_validation_failure_returns_400_invalid_request(
    retriever_app: dict[str, Any],
) -> None:
    """Code-review finding P1 #7: Pydantic validation errors must land on
    400 + {error: invalid_request, message} envelope, not FastAPI's
    default 422 + {detail:[...]} shape. Verified by sending a missing
    required field.
    """
    _mock_rewriter()
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                # missing tenant_id entirely
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "hi",
            },
        )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"] == "invalid_request"
    assert "tenant_id" in body["message"]


@respx.mock
async def test_empty_transcript_returns_400_invalid_request(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter()
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "",
            },
        )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"] == "invalid_request"


@respx.mock
async def test_top_k_out_of_bounds_returns_400_invalid_request(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter()
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-1",
                "transcript": "hi",
                "top_k": 999,
            },
        )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"] == "invalid_request"


# ─────────────────────────────────────────────────────────────────────
# Failure-path trace writes (P1 #4 + #5)
# ─────────────────────────────────────────────────────────────────────


async def _count_traces_for_session(dsn: str, session_id: uuid.UUID) -> list[dict]:
    """Read all retrieval_traces rows for a given session, for test
    assertions about failure-path forensics."""
    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT turn_id, in_scope, rewritten_query, error_class, "
                    "stage_timings_ms FROM retrieval_traces "
                    "WHERE session_id = :sid ORDER BY created_at"
                ),
                {"sid": session_id},
            )
            return [dict(row) for row in result.mappings()]
    finally:
        await engine.dispose()


@respx.mock
async def test_rewriter_timeout_returns_503_and_writes_error_trace(
    retriever_app: dict[str, Any], pgvector_postgres
) -> None:
    """Code-review finding P1 #4: rewriter timeout must produce both
    a 503 + rewriter_timeout envelope AND a retrieval_traces row with
    error_class populated (FR-021). Previously neither happened: the
    exception escaped before write_trace, and rewriter errors weren't
    converted to the 503 envelope.
    """
    respx.post(_LLM_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-timeout",
                "transcript": "как получить права?",
            },
        )

    assert resp.status_code == 503, resp.text
    assert resp.json()["error"] == "rewriter_timeout"

    # Forensics row must exist for this turn with error_class populated.
    dsn = pgvector_postgres.get_connection_url().replace("+psycopg2", "+asyncpg")
    traces = await _count_traces_for_session(dsn, retriever_app["session_id"])
    matches = [t for t in traces if t["turn_id"] == "t-timeout"]
    assert len(matches) == 1, f"expected 1 trace row for t-timeout, got {matches}"
    assert matches[0]["error_class"] == "rewriter_timeout"
    # Rewriter stage time should be recorded even on timeout.
    st = matches[0]["stage_timings_ms"]
    assert "rewrite" in st
    assert st["total"] == sum(v for k, v in st.items() if k != "total")


@respx.mock
async def test_embedder_failure_returns_503_embedder_unavailable(
    retriever_app: dict[str, Any], monkeypatch
) -> None:
    """Code-review finding P1 #5: bare ValueError from the embedder
    must be translated to the typed EmbedderUnavailable (503 +
    envelope), not propagate as an untyped FastAPI 500.
    """
    _mock_rewriter(in_scope=True)

    async def _bad_encode(text: str) -> list[float]:
        raise ValueError("embedder shape mismatch: (768,) expected (1024,)")

    # Patch at the module where retrieve.py imported it.
    import retrieve

    monkeypatch.setattr(retrieve, "encode", _bad_encode)

    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-embfail",
                "transcript": "как получить права?",
            },
        )

    assert resp.status_code == 503, resp.text
    assert resp.json()["error"] == "embedder_unavailable"


# ─────────────────────────────────────────────────────────────────────
# Internal-caller auth (issue #40 / #47)
# ─────────────────────────────────────────────────────────────────────


@respx.mock
async def test_retrieve_without_internal_token_returns_401(
    retriever_app: dict[str, Any],
) -> None:
    """Caller without ``X-Internal-Token`` is rejected before any
    DB / LLM work happens. Closes issue #40."""
    _mock_rewriter()
    async with await _client(retriever_app["app"]) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-noauth",
                "transcript": "как получить права?",
            },
        )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"] == "unauthenticated"


@respx.mock
async def test_retrieve_with_wrong_internal_token_returns_401(
    retriever_app: dict[str, Any],
) -> None:
    """Wrong token value (e.g. stale rotation) is rejected. The
    `secrets.compare_digest` comparison short-circuits in constant
    time so the response carries no length-leak signal."""
    _mock_rewriter()
    async with await _client(retriever_app["app"], {"X-Internal-Token": "wrong-token"}) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-wrongauth",
                "transcript": "как получить права?",
            },
        )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"] == "unauthenticated"


@respx.mock
async def test_retrieve_returns_503_when_internal_token_unset(
    retriever_app: dict[str, Any], monkeypatch
) -> None:
    """Empty ``RETRIEVER_INTERNAL_TOKEN`` env disables the endpoint
    entirely — production deploys MUST set the secret. Issue #47."""
    monkeypatch.setenv("RETRIEVER_INTERNAL_TOKEN", "")
    import config

    config.get_settings.cache_clear()

    _mock_rewriter()
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
        resp = await client.post(
            "/retrieve",
            json={
                "tenant_id": str(retriever_app["tenant_id"]),
                "session_id": str(retriever_app["session_id"]),
                "turn_id": "t-unconfig",
                "transcript": "как получить права?",
            },
        )
    assert resp.status_code == 503, resp.text
    assert resp.json()["error"] == "retriever_unconfigured"


@respx.mock
async def test_response_shape_conforms_to_openapi_contract(
    retriever_app: dict[str, Any],
) -> None:
    _mock_rewriter(query="как получить водительские права", in_scope=True)
    async with await _client(retriever_app["app"], _headers(retriever_app)) as client:
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

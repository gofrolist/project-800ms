"""Unit tests for `services/retriever/rewriter.py`.

Mocks the LLM HTTP call via `respx` — no live endpoint, no DB, no
container. Fast enough to run on every pre-push.

Covers the four T019 properties:

  (a) well-formed JSON is parsed into `{query, in_scope}`
  (b) malformed JSON triggers `RewriterMalformedOutput`
  (c) history is bounded to the last 6 entries before reaching the LLM
  (d) the `model` arg is threaded through into the request body
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from errors import RewriterMalformedOutput, RewriterTimeout
from rewriter import REWRITER_VERSION, RewriterResult, rewrite_and_classify

_LLM_BASE = "http://llm-test.local"
_LLM_URL = f"{_LLM_BASE}/chat/completions"


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the rewriter at the mock LLM host per-test.

    Clears the config singleton so each test re-reads env; autouse so
    we can't forget it. The DB_URL / LIVEKIT_* envs still come from the
    outer test-runner command.
    """
    monkeypatch.setenv("LLM_BASE_URL", _LLM_BASE)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("REWRITER_MODEL", "default-model")
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _llm_json(content: str) -> dict:
    """Shape the canonical OpenAI-compatible chat response envelope."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@respx.mock
async def test_parses_well_formed_json_into_result() -> None:
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "как получить водительские права", "in_scope": true}'),
        )
    )
    result = await rewrite_and_classify(
        "как получить ПРАВА",
        history=[],
        model="test-model",
    )
    assert isinstance(result, RewriterResult)
    assert result.query == "как получить водительские права"
    assert result.in_scope is True


@respx.mock
async def test_malformed_content_raises_malformed_output() -> None:
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json("sorry, I can't comply with this request"),
        )
    )
    with pytest.raises(RewriterMalformedOutput):
        await rewrite_and_classify("hi", history=[], model="test-model")


@respx.mock
async def test_missing_in_scope_field_raises_malformed_output() -> None:
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "only this field present"}'),
        )
    )
    with pytest.raises(RewriterMalformedOutput):
        await rewrite_and_classify("hi", history=[], model="test-model")


@respx.mock
async def test_wrong_field_types_raise_malformed_output() -> None:
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": 42, "in_scope": "yes"}'),
        )
    )
    with pytest.raises(RewriterMalformedOutput):
        await rewrite_and_classify("hi", history=[], model="test-model")


@respx.mock
async def test_non_2xx_raises_malformed_output() -> None:
    # Rewriter treats any non-2xx as unusable (callers fail-closed to
    # refusal). 500, 401, 429 all surface the same way here; the
    # deployed /ready probe distinguishes the auth codes for alerting.
    respx.post(_LLM_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(RewriterMalformedOutput):
        await rewrite_and_classify("hi", history=[], model="test-model")


@respx.mock
async def test_timeout_raises_rewriter_timeout() -> None:
    respx.post(_LLM_URL).mock(side_effect=httpx.ReadTimeout("upstream slow"))
    with pytest.raises(RewriterTimeout):
        await rewrite_and_classify("hi", history=[], model="test-model")


@respx.mock
async def test_history_is_bounded_to_last_6_entries() -> None:
    """History > 6 entries must be truncated to the most recent 6 before
    reaching the LLM. Older entries must NOT appear in the request body.
    """
    route = respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "ok", "in_scope": true}'),
        )
    )
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "text": f"turn-{i:02d}"} for i in range(10)
    ]
    await rewrite_and_classify("current", history=history, model="test-model")

    assert route.called
    sent_body = json.loads(route.calls[0].request.content)
    sent_contents = [m["content"] for m in sent_body["messages"]]
    joined = " ".join(sent_contents)

    # First 4 of 10 must be dropped (we keep the last 6).
    for i in range(4):
        assert f"turn-{i:02d}" not in joined, f"older history entry turn-{i:02d} leaked into prompt"
    # Last 6 must be present.
    for i in range(4, 10):
        assert f"turn-{i:02d}" in joined, f"recent entry turn-{i:02d} missing"


@respx.mock
async def test_model_arg_threaded_into_request_body() -> None:
    route = respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "q", "in_scope": true}'),
        )
    )
    await rewrite_and_classify("hi", history=[], model="my-custom-model-v9")
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "my-custom-model-v9"


@respx.mock
async def test_rewriter_version_is_a_stable_constant() -> None:
    # Traces store REWRITER_VERSION so a past trace can be reproduced
    # against the exact prompt version that wrote it. Changing this is
    # a breaking change for historic replays and must be a PR bump.
    assert isinstance(REWRITER_VERSION, str)
    assert REWRITER_VERSION.startswith("rewriter-")


@respx.mock
async def test_transcript_is_the_final_user_message() -> None:
    route = respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "q", "in_scope": true}'),
        )
    )
    await rewrite_and_classify(
        "как получить права?",
        history=[{"role": "user", "text": "привет"}],
        model="test-model",
    )
    body = json.loads(route.calls[0].request.content)
    messages = body["messages"]
    # system → history → current transcript as the final user message.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "как получить права?"


@respx.mock
async def test_out_of_scope_classification_returns_false() -> None:
    respx.post(_LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json=_llm_json('{"query": "погода в Москве", "in_scope": false}'),
        )
    )
    result = await rewrite_and_classify("какая сейчас погода?", history=[], model="test-model")
    assert result.in_scope is False

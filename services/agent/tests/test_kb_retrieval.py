"""Unit tests for `services/agent/kb_retrieval.py` — KBRetrievalProcessor.

Uses respx to mock the /retrieve HTTP call. Captures frames pushed by
the processor via a monkey-patched `push_frame`. No real pipecat
pipeline, no real retriever.

Covers the four T021 properties:
  (a) TranscriptionFrame triggers a /retrieve call with the expected
      payload shape (tenant_id, session_id, turn_id, transcript).
  (b) An in-scope response pushes an LLMMessagesAppendFrame carrying
      the grounded system prompt and the rewritten query.
  (c) A retriever 503 falls back to the refusal path without crashing
      the pipeline — still pushes an LLMMessagesAppendFrame, never
      raises.
  (d) `turn_id` increments monotonically within the processor
      instance's lifetime.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
import pytest
import respx
from pipecat.frames.frames import LLMMessagesAppendFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from kb_prompts import (
    REFUSAL_SYSTEM_PROMPT_RU,
    GROUNDED_SYSTEM_PROMPT_RU,
)
from kb_retrieval import KBRetrievalProcessor

_RETRIEVER_BASE = "http://test-retriever.local"


def _response_body(
    *,
    in_scope: bool = True,
    rewritten: str = "как получить права",
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "rewritten_query": rewritten,
        "in_scope": in_scope,
        "chunks": chunks
        if chunks is not None
        else [
            {
                "id": 42,
                "title": "Водительские права — Получение",
                "content": "Чтобы получить права, посетите автошколу.",
                "score": 0.9,
                "fusion_components": {"semantic": 0.8, "lexical": 0.5},
                "metadata": {},
            }
        ],
        "stage_timings_ms": {"rewrite": 50, "embed": 20, "sql": 10, "total": 80},
        "trace_id": str(uuid.uuid4()),
    }


@pytest.fixture
def captured() -> list[tuple[object, object]]:
    return []


@pytest.fixture
def processor(captured, monkeypatch) -> KBRetrievalProcessor:
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    proc = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE,
        tenant_id=tenant_id,
        session_id=session_id,
        timeout_ms=500,
    )
    proc._test_tenant_id = tenant_id
    proc._test_session_id = session_id

    async def _capture(frame: object, direction: object = FrameDirection.DOWNSTREAM) -> None:
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    return proc


def _tx_frame(text: str) -> TranscriptionFrame:
    """Build a minimally-valid TranscriptionFrame."""
    return TranscriptionFrame(text=text, user_id="u", timestamp="2026-04-24T00:00:00Z")


@respx.mock
async def test_transcription_triggers_retrieve_call_with_expected_payload(
    processor, captured
) -> None:
    route = respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(200, json=_response_body())
    )
    frame = _tx_frame("как получить ПРАВА")
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert route.called, "expected /retrieve to be called on TranscriptionFrame"
    body = json.loads(route.calls[0].request.content)
    assert body["tenant_id"] == str(processor._test_tenant_id)
    assert body["session_id"] == str(processor._test_session_id)
    assert body["transcript"] == "как получить ПРАВА"
    # Format is `<4-hex-salt>-<5+-digit-counter>` per issue #48. The
    # `\d{5,}` (not `\d{5}`) tolerates the >99,999-turn case in case
    # one process accidentally lives that long.
    assert re.match(r"^[0-9a-f]{4}-\d{5,}$", body["turn_id"])


@respx.mock
async def test_outgoing_request_carries_x_internal_token_when_set(captured, monkeypatch) -> None:
    """Issue #40: the agent MUST attach `X-Internal-Token` to every
    /retrieve POST when the secret is configured. A regression that
    drops the header silently turns every turn into a refusal (401
    on the retriever side) — feature breaks but tests stay green
    unless we pin this assertion (review finding TEST-001).
    """
    proc = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE,
        tenant_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        internal_token="secret-token-xyz",
    )

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    route = respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(200, json=_response_body())
    )
    await proc.process_frame(_tx_frame("вопрос"), FrameDirection.DOWNSTREAM)

    assert route.called
    headers = route.calls[0].request.headers
    assert headers.get("x-internal-token") == "secret-token-xyz"


@respx.mock
async def test_outgoing_request_omits_x_internal_token_when_unset(captured, monkeypatch) -> None:
    """The complementary contract: empty `internal_token` must NOT
    attach the header. The retriever returns 401 on missing header,
    which surfaces as a kb_retrieval.auth_failed log on-call can
    triage."""
    proc = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE,
        tenant_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        internal_token="",  # empty
    )

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    route = respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(401, json={"error": "unauthenticated"})
    )
    await proc.process_frame(_tx_frame("вопрос"), FrameDirection.DOWNSTREAM)

    assert route.called
    headers = route.calls[0].request.headers
    assert "x-internal-token" not in {k.lower() for k in headers}


@respx.mock
async def test_in_scope_result_pushes_llm_messages_append_with_grounded_prompt(
    processor, captured
) -> None:
    respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(200, json=_response_body(in_scope=True))
    )
    await processor.process_frame(_tx_frame("как получить права?"), FrameDirection.DOWNSTREAM)

    pushed_frames = [f for f, _ in captured]
    # The original TranscriptionFrame is SWALLOWED — only our composed
    # LLMMessagesAppendFrame should reach downstream.
    assert len(pushed_frames) == 1
    assert isinstance(pushed_frames[0], LLMMessagesAppendFrame)

    # The grounded system prompt is present in the first system message.
    msgs = pushed_frames[0].messages
    assert msgs[0]["role"] == "system"
    assert GROUNDED_SYSTEM_PROMPT_RU.split(".")[0] in msgs[0]["content"]
    # The retrieved chunk's title appears in the context block.
    assert "Водительские права — Получение" in msgs[0]["content"]
    # The user message is the REWRITTEN query, not the raw transcript.
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "как получить права"
    # run_llm=True is LOAD-BEARING: pipecat's aggregator only invokes
    # the LLM when this flag is truthy (see _handle_llm_messages_append
    # in llm_response_universal.py). Without it the grounded context
    # reaches memory but the LLM never fires — feature silently inert.
    assert pushed_frames[0].run_llm is True


@respx.mock
async def test_retriever_503_falls_back_to_refusal_without_crashing(processor, captured) -> None:
    respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(503, json={"error": "db_unavailable"})
    )
    # Must NOT raise.
    await processor.process_frame(_tx_frame("как получить права?"), FrameDirection.DOWNSTREAM)

    pushed_frames = [f for f, _ in captured]
    assert len(pushed_frames) == 1
    assert isinstance(pushed_frames[0], LLMMessagesAppendFrame)
    # Refusal prompt, not grounded.
    msgs = pushed_frames[0].messages
    assert REFUSAL_SYSTEM_PROMPT_RU.split(".")[0] in msgs[0]["content"]
    # Refusal path must also trigger the LLM — otherwise the user hears
    # nothing after an out-of-scope/error turn.
    assert pushed_frames[0].run_llm is True


@respx.mock
async def test_retriever_timeout_falls_back_to_refusal(processor, captured) -> None:
    respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(side_effect=httpx.ReadTimeout("upstream slow"))
    await processor.process_frame(_tx_frame("как получить права?"), FrameDirection.DOWNSTREAM)

    pushed_frames = [f for f, _ in captured]
    assert len(pushed_frames) == 1
    assert isinstance(pushed_frames[0], LLMMessagesAppendFrame)
    msgs = pushed_frames[0].messages
    assert REFUSAL_SYSTEM_PROMPT_RU.split(".")[0] in msgs[0]["content"]


@respx.mock
async def test_out_of_scope_response_falls_back_to_refusal(processor, captured) -> None:
    respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(200, json=_response_body(in_scope=False, chunks=[]))
    )
    await processor.process_frame(_tx_frame("какая погода?"), FrameDirection.DOWNSTREAM)

    pushed_frames = [f for f, _ in captured]
    assert len(pushed_frames) == 1
    assert isinstance(pushed_frames[0], LLMMessagesAppendFrame)
    msgs = pushed_frames[0].messages
    assert REFUSAL_SYSTEM_PROMPT_RU.split(".")[0] in msgs[0]["content"]


@respx.mock
async def test_turn_id_monotonic_within_session(processor, captured) -> None:
    """Issue #48: turn_id format is `<instance-salt>-<NNNNN>`. The
    counter is monotonic per processor; the salt is randomized in
    __init__ so a restart inside the same session can't replay
    earlier turn_ids and collide with UNIQUE(session_id, turn_id)."""
    route = respx.post(f"{_RETRIEVER_BASE}/retrieve").mock(
        return_value=httpx.Response(200, json=_response_body())
    )
    for _ in range(3):
        await processor.process_frame(_tx_frame("вопрос"), FrameDirection.DOWNSTREAM)

    assert route.call_count == 3
    turn_ids = [json.loads(call.request.content)["turn_id"] for call in route.calls]
    pattern = re.compile(r"^[0-9a-f]{4}-\d{5,}$")
    for tid in turn_ids:
        assert pattern.match(tid), tid
    # The numeric suffix is monotonic 1..3.
    suffixes = [int(t.split("-", 1)[1]) for t in turn_ids]
    assert suffixes == [1, 2, 3]
    # All three share the same instance salt — same processor instance.
    salts = {t.split("-", 1)[0] for t in turn_ids}
    assert len(salts) == 1


def test_turn_id_salt_differs_across_processor_instances() -> None:
    """Two processors built in the same session MUST get distinct
    salts so one's counter can never alias onto the other's. This
    is the post-restart collision guard from issue #48."""
    p1 = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE, tenant_id=uuid.uuid4(), session_id=uuid.uuid4()
    )
    p2 = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE, tenant_id=uuid.uuid4(), session_id=uuid.uuid4()
    )
    # 16-bit salt collision is ~1/65536 — astronomically unlikely
    # given test count, so we can assert flat inequality.
    assert p1._instance_salt != p2._instance_salt


async def test_empty_transcript_is_forwarded_untouched(captured, monkeypatch) -> None:
    """Empty STT output isn't a user turn — don't waste a retriever call."""
    proc = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE,
        tenant_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
    )

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    frame = _tx_frame("   ")
    await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

    # Empty → frame passes through unchanged.
    assert len(captured) == 1
    assert captured[0][0] is frame


async def test_pass_through_mode_when_tenant_missing(captured, monkeypatch) -> None:
    """tenant_id=None or session_id=None disables the processor (pass-through)."""
    proc = KBRetrievalProcessor(
        retriever_url=_RETRIEVER_BASE,
        tenant_id=None,
        session_id=uuid.uuid4(),
    )

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    frame = _tx_frame("любой запрос")
    await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

    # No retriever call, frame forwarded as-is.
    assert len(captured) == 1
    assert captured[0][0] is frame
    assert not proc.is_enabled
    assert proc._disable_reason() == "tenant_id_missing"


async def test_pass_through_mode_when_retriever_url_empty(captured, monkeypatch) -> None:
    """Empty retriever_url must also route the processor to pass-through.

    Regression guard for code-review finding P1 #3: the earlier sentinel
    URL `http://disabled` combined with "is_enabled only checks
    tenant/session" caused every turn to DNS-fail and hit the refusal
    branch — the opposite of pass-through.
    """
    proc = KBRetrievalProcessor(
        retriever_url="",
        tenant_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
    )

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    monkeypatch.setattr(proc, "push_frame", _capture)
    frame = _tx_frame("любой запрос")
    await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

    # No retriever call, frame forwarded as-is (NOT routed to refusal).
    assert len(captured) == 1
    assert captured[0][0] is frame
    assert not proc.is_enabled
    assert proc._disable_reason() == "retriever_url_missing"

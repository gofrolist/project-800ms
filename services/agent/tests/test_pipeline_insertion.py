"""Regression guard on the processor ordering in `pipeline.py`.

The KBRetrievalProcessor MUST sit between `user_transcript` and
`user_agg`. Moving it anywhere else breaks the RAG grounding flow:

* Before `user_transcript` — the forwarder never captures finalized
  transcripts, so the web UI doesn't render them.
* After `user_agg` — the aggregator has already flushed the user turn
  to the LLM without RAG context, so the grounded prompt is too late.
* After `llm` — the LLMMessagesAppendFrame lands in a context that's
  already been sent to the model.

Rather than build a live pipeline (expensive: needs GigaAM + LLM +
TTS), this test parses `pipeline.py` source and asserts the literal
processor list ordering. A source-level check catches the exact class
of regression (someone reshuffles the list by accident) without paying
the pipecat runtime cost.
"""

from __future__ import annotations

import re
from pathlib import Path

_PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"

# Expected processor ordering inside `Pipeline([...])`. The parenthesized
# suffixes (e.g. `()`) match the source text.
_EXPECTED_ORDER: list[str] = [
    "transport.input()",
    "stt",
    "user_transcript",
    "kb_retrieval",
    "user_agg",
    "llm",
    "assistant_transcript",
    "tts",
    "error_forwarder",
    "transport.output()",
    "assistant_agg",
]


def _extract_pipeline_processors() -> list[str]:
    """Return the list of processor tokens that appear inside the
    `Pipeline([...])` literal in `pipeline.py`, in order.

    Whitespace and trailing commas are stripped. Comment lines (`#
    ...`) are ignored so a future reviewer can annotate the block
    without breaking this test.
    """
    src = _PIPELINE_PATH.read_text(encoding="utf-8")
    match = re.search(r"pipeline\s*=\s*Pipeline\(\s*\[(.*?)\]\s*\)", src, re.DOTALL)
    assert match is not None, "couldn't locate `pipeline = Pipeline([...])` in pipeline.py"

    body = match.group(1)
    tokens: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip trailing comma(s) and inline comments
        token = line.split("#", 1)[0].rstrip().rstrip(",").strip()
        if token:
            tokens.append(token)
    return tokens


def test_kb_retrieval_sits_between_user_transcript_and_user_agg() -> None:
    tokens = _extract_pipeline_processors()
    assert "kb_retrieval" in tokens, (
        "KBRetrievalProcessor was removed from the pipeline — RAG grounding "
        "is disabled for every session."
    )
    kb_idx = tokens.index("kb_retrieval")
    assert tokens[kb_idx - 1] == "user_transcript", (
        f"kb_retrieval must follow user_transcript; got predecessor {tokens[kb_idx - 1]!r}"
    )
    assert tokens[kb_idx + 1] == "user_agg", (
        f"kb_retrieval must precede user_agg; got successor {tokens[kb_idx + 1]!r}"
    )


def test_full_pipeline_ordering_matches_spec() -> None:
    """Full ordering assertion. Updates here require a paired update in
    `_EXPECTED_ORDER` and a PR reviewer sign-off — the list is the
    documented contract for the voice pipeline topology."""
    tokens = _extract_pipeline_processors()
    assert tokens == _EXPECTED_ORDER, (
        f"pipeline processor ordering changed\n  expected: {_EXPECTED_ORDER}\n  actual:   {tokens}"
    )

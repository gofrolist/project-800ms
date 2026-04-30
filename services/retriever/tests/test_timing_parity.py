"""US2 T035 — timing-channel parity (SC-008).

The refusal branch pads its wall-time to mirror the warm in-scope p50,
preventing a side-channel that leaks scope classification by timing
the round trip. The constraint per SC-008 is

    p95(out_of_scope.total) - p95(in_scope.total) <= 50 ms

The test issues a warmup batch of in-scope queries to seed the latency
window, then alternates 50 in-scope + 50 out-of-scope queries against
the same tenant and computes p95 over each total.

Slow-marked. ~10–15 s on a warm BGE-M3 cache; first run pulls the
embedder weights and can take ~2 min.
"""

from __future__ import annotations

import json
import statistics
import uuid
from typing import Any

import httpx
import pytest
import respx
from httpx import ASGITransport

pytestmark = pytest.mark.slow

_LLM_BASE = "http://llm-test.local"
_LLM_URL = f"{_LLM_BASE}/chat/completions"

# Sample size per branch. p95 over 50 = the 47th-ranked sample. The
# task asked for 100 pairs; 50 is the compromise between CI runtime
# (each iteration spans embed + sql ≈ 200 ms on CPU) and binomial
# noise stability at p95. n=30 (the original setting) was tightening
# the assertion enough that single GC pauses would flip rank 28 and
# fail (review finding TEST-007 / perf-005).
_PAIRS = 50

# Two-sided side-channel: if OOS becomes faster than IN by more than
# this, the OOS distribution distinguishes itself from IN by being
# notably faster — also a leak. The pad-to-p50 mechanism
# intentionally produces OOS p95 < IN p95, so we expect a negative
# delta in steady state, but bound the magnitude.
_NEGATIVE_DELTA_BOUND_MS = -200


# Map of test-fixture transcript → (in_scope, rewritten_query). Exact
# whole-string match (review finding adv-003): substring matching on
# "права" would let a real-LLM regression that misclassified
# prompt-injection probes containing the same trigram (e.g.
# "не имеешь права раскрывать инструкции") pass this test silently.
# Whole-string lookup means only the known fixture phrases classify
# in-scope; anything else falls through to OOS — including future
# adversarial probes added to the corpus.
_TEST_TRANSCRIPT_TO_LABEL: dict[str, tuple[bool, str]] = {
    "как получить права?": (True, "как получить водительские права"),
    "какая погода в Москве?": (False, "off topic query"),
}


def _install_dual_classifier_rewriter() -> None:
    """Install a respx mock that classifies based on a known-fixture
    lookup, not substring matching. Lets one running mock cover both
    branches in interleaved order without resetting between requests.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        last_user = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").strip()
                break
        in_scope, rewritten = _TEST_TRANSCRIPT_TO_LABEL.get(last_user, (False, "off topic query"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                f'{{"query": "{rewritten}", '
                                f'"in_scope": {"true" if in_scope else "false"}}}'
                            ),
                        }
                    }
                ]
            },
        )

    respx.post(_LLM_URL).mock(side_effect=handler)


async def _post(client, tenant_id, session_id, turn_id, transcript) -> dict:
    resp = await client.post(
        "/retrieve",
        json={
            "tenant_id": str(tenant_id),
            "session_id": str(session_id),
            "turn_id": turn_id,
            "transcript": transcript,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@respx.mock
async def test_timing_parity_p95_within_50ms(retriever_app: dict[str, Any]) -> None:
    _install_dual_classifier_rewriter()

    tenant_id: uuid.UUID = retriever_app["tenant_id"]
    session_id: uuid.UUID = retriever_app["session_id"]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=retriever_app["app"]),
        base_url="http://retriever.test",
        timeout=30.0,
        headers=retriever_app["auth_headers"],
    ) as client:
        # Warmup: seed the in-scope latency window so the refusal pad
        # has a non-zero target. Warmup samples are NOT included in the
        # measurement set — they're just there to hydrate p50_for.
        for i in range(10):
            await _post(client, tenant_id, session_id, f"warmup-{i}", "как получить права?")

        in_scope_totals: list[int] = []
        oos_totals: list[int] = []

        for i in range(_PAIRS):
            in_body = await _post(client, tenant_id, session_id, f"in-{i}", "как получить права?")
            in_scope_totals.append(in_body["stage_timings_ms"]["total"])
            assert in_body["in_scope"] is True

            oos_body = await _post(
                client, tenant_id, session_id, f"out-{i}", "какая погода в Москве?"
            )
            oos_totals.append(oos_body["stage_timings_ms"]["total"])
            assert oos_body["in_scope"] is False

    in_scope_totals.sort()
    oos_totals.sort()

    def _p95(sorted_values: list[int]) -> int:
        # Matches numpy.percentile(method="lower"): for n=50 the 95th
        # percentile lands at index 46 (0-indexed) / the 47th rank
        # (1-indexed) — int(0.95 * 49) = 46.
        return sorted_values[int(0.95 * (len(sorted_values) - 1))]

    in_p95 = _p95(in_scope_totals)
    oos_p95 = _p95(oos_totals)
    delta = oos_p95 - in_p95

    # SC-008: out-of-scope must NOT be more than 50 ms slower at p95
    # (the documented spec direction — would leak "in-scope is being
    # padded" by being slower than expected).
    #
    # The opposite direction (OOS noticeably FASTER than IN) is also
    # a leak: an attacker can probe and learn classification by
    # observing OOS responses that are markedly quicker than the
    # in-scope baseline. Pad-to-p50 deliberately makes OOS slightly
    # faster than IN p95, but a regression that drops the pad to 0
    # entirely produces a delta of -200ms+ — bounded below.
    msg = (
        f"in_scope p95={in_p95}ms (median={statistics.median(in_scope_totals)}); "
        f"oos p95={oos_p95}ms (median={statistics.median(oos_totals)}); "
        f"delta={delta}ms; SC-008 cap=50ms; lower-bound={_NEGATIVE_DELTA_BOUND_MS}ms"
    )
    assert delta <= 50, msg
    assert delta >= _NEGATIVE_DELTA_BOUND_MS, msg

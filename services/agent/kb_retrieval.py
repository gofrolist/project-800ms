"""KBRetrievalProcessor — bridges the Pipecat pipeline to the retriever HTTP service.

Sits between `user_transcript` and `user_agg` (see `pipeline.py` line
210). On each finalized `TranscriptionFrame`, the processor:

1. Increments a monotonic `turn_id`.
2. POSTs to `{retriever_url}/retrieve` with `(tenant_id, session_id,
   turn_id, transcript)`.
3. On 200 + `in_scope=true`: pushes an `LLMMessagesAppendFrame` carrying
   the grounded system prompt + the retrieved context + the rewriter's
   `rewritten_query` (as a user message).
4. On `in_scope=false` / 5xx / timeout / any error: pushes an
   `LLMMessagesAppendFrame` with the refusal fallback so the LLM still
   has SOMETHING to answer with. Never crashes the pipeline.
5. SWALLOWS the original `TranscriptionFrame` on both paths — the
   aggregator should see the rewritten query (grounded) or the raw
   transcript (refusal), NOT both. Forwarding it would give the LLM
   two "user turns" in the same conversational beat.

Pass-through mode: if `tenant_id` or `session_id` is unset, the
processor forwards every frame untouched. Lets operators ship the
retriever service before all dispatch-path tenancy plumbing lands
without breaking non-KB voice sessions.

Blocking model: `process_frame` awaits the retriever call before
pushing any downstream frames. The retriever SLO bounds the wait (≤800
ms p95 per constitution Principle I); during that window the pipeline
is paused, which is preferable to the alternative (async fire-and-
forget would race the aggregator's VAD-driven flush).
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from kb_prompts import build_grounded_messages, build_refusal_messages

# Default request timeout. Matches the AGENT_RETRIEVER_TIMEOUT_MS env
# the operator sets. Larger than the retriever's own rewriter_timeout_ms
# so the retriever can still fail-closed to refusal and return a
# structured 200 before the agent side gives up.
_DEFAULT_TIMEOUT_MS = 2000


class KBRetrievalProcessor(FrameProcessor):
    """Pipecat FrameProcessor that calls the retriever on each turn.

    Args:
        retriever_url: Base URL of the retriever service (e.g.
            `http://retriever:8002`). Trailing slash tolerated.
        tenant_id: Tenant UUID. When None, the processor becomes a
            pass-through — lets the agent boot in deploys that haven't
            plumbed tenant routing yet.
        session_id: Sessions table row UUID. Must match the value the
            API used when creating the sessions row (retrieval_traces
            FKs to sessions.id). None → pass-through.
        timeout_ms: Hard HTTP timeout. A stalled retriever must not
            hang the entire voice pipeline; the refusal-fallback path
            fires on timeout.
    """

    def __init__(
        self,
        *,
        retriever_url: str,
        tenant_id: uuid.UUID | None,
        session_id: uuid.UUID | None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._retriever_url = retriever_url.rstrip("/")
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._timeout_ms = timeout_ms
        self._turn_counter = 0
        # Shared httpx client reused across every turn — one DNS +
        # TCP + TLS setup for the whole session instead of per turn.
        # Initialized lazily in enabled mode; stays None in pass-
        # through mode so we never pay for an unused pool.
        # Code-review finding P1 #8.
        self._client: httpx.AsyncClient | None = None

        # Single mode evaluation at construction so the operator gets
        # one log line instead of one-per-turn, and misconfiguration
        # ("tenant_id set but retriever_url missing") is visible at boot.
        if not self.is_enabled:
            reason = self._disable_reason()
            logger.warning(
                "kb_retrieval.passthrough_mode reason={reason} "
                "tenant_set={tenant_set} session_set={session_set} "
                "retriever_url_set={url_set}",
                reason=reason,
                tenant_set=tenant_id is not None,
                session_set=session_id is not None,
                url_set=bool(self._retriever_url),
            )
        else:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_ms / 1000, connect=1.0),
            )

    async def cleanup(self) -> None:
        """Release the shared httpx client on pipeline teardown."""
        await super().cleanup()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def is_enabled(self) -> bool:
        """False when any of retriever_url / tenant_id / session_id is missing.

        Previously the check was tenant/session only — the processor
        relied on a sentinel `http://disabled` retriever URL that would
        DNS-fail and route every turn through the except-→-refusal
        path. That contradicted the pass-through contract. Every leg
        that makes the HTTP call unsafe must drop the processor to
        pass-through so the raw TranscriptionFrame reaches the
        aggregator unchanged.
        """
        return (
            bool(self._retriever_url)
            and self._tenant_id is not None
            and self._session_id is not None
        )

    def _disable_reason(self) -> str:
        """Human-readable reason for pass-through mode, for logs + tests."""
        if not self._retriever_url:
            return "retriever_url_missing"
        if self._tenant_id is None:
            return "tenant_id_missing"
        if self._session_id is None:
            return "session_id_missing"
        return "enabled"

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Only intercept finalized user transcriptions going DOWNSTREAM
        # (toward the aggregator + LLM). Interim frames and upstream
        # frames (e.g. cancellation) pass through untouched.
        if (
            not self.is_enabled
            or not isinstance(frame, TranscriptionFrame)
            or direction != FrameDirection.DOWNSTREAM
        ):
            await self.push_frame(frame, direction)
            return

        transcript = (frame.text or "").strip()
        if not transcript:
            # Empty STT output — nothing to retrieve against. Let the
            # aggregator see the frame and decide what to do.
            await self.push_frame(frame, direction)
            return

        self._turn_counter += 1
        turn_id = f"t-{self._turn_counter:05d}"

        try:
            messages = await self._retrieve_and_compose(transcript, turn_id)
        except Exception as exc:  # noqa: BLE001 — logged + recovered via refusal
            logger.warning(
                "kb_retrieval.call_failed turn={turn} kind={kind}",
                turn=turn_id,
                kind=type(exc).__name__,
            )
            messages = build_refusal_messages(transcript)

        # Swallow the original TranscriptionFrame. The aggregator will
        # receive our composed messages instead; forwarding the raw
        # transcript here would give the LLM two user turns (raw +
        # rewritten), which confuses the grounded-answer contract.
        #
        # run_llm=True is LOAD-BEARING. Without it, pipecat's
        # LLMContextAggregatorPair._handle_llm_messages_append only
        # calls self.add_messages(...) and skips push_context_frame —
        # meaning our composed context reaches memory but the LLM is
        # never invoked. Combined with swallowing the transcript
        # (which would have been the aggregator's normal trigger),
        # the grounded-answer path becomes silently inert. See
        # services/agent/.venv/…/pipecat/processors/aggregators/
        # llm_response_universal.py:636-644.
        await self.push_frame(
            LLMMessagesAppendFrame(messages=messages, run_llm=True),
            direction,
        )

    async def _retrieve_and_compose(self, transcript: str, turn_id: str) -> list[dict[str, str]]:
        """POST /retrieve and return the LLMMessagesAppendFrame payload.

        Returns either the grounded messages (on in_scope=true + chunks)
        or the refusal messages (on in_scope=false, empty chunks, or
        non-2xx). Raises on transport / timeout so the caller's except
        handler logs it and falls back.
        """
        # tenant_id / session_id / client are checked non-None by is_enabled.
        assert self._tenant_id is not None
        assert self._session_id is not None
        assert self._client is not None

        payload = {
            "tenant_id": str(self._tenant_id),
            "session_id": str(self._session_id),
            "turn_id": turn_id,
            "transcript": transcript,
        }

        resp = await self._client.post(
            f"{self._retriever_url}/retrieve",
            json=payload,
        )

        if resp.status_code != 200:
            logger.warning(
                "kb_retrieval.non_200 turn={turn} status={status}",
                turn=turn_id,
                status=resp.status_code,
            )
            return build_refusal_messages(transcript)

        body = resp.json()
        if not body.get("in_scope") or not body.get("chunks"):
            logger.info(
                "kb_retrieval.refusal turn={turn} reason={reason}",
                turn=turn_id,
                reason="out_of_scope" if not body.get("in_scope") else "empty_chunks",
            )
            return build_refusal_messages(transcript)

        return build_grounded_messages(
            rewritten_query=body.get("rewritten_query") or transcript,
            chunks=body["chunks"],
        )

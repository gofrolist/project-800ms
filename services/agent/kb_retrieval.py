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

    @property
    def is_enabled(self) -> bool:
        """False when tenant_id or session_id is missing (pass-through mode)."""
        return self._tenant_id is not None and self._session_id is not None

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
        await self.push_frame(
            LLMMessagesAppendFrame(messages=messages),
            direction,
        )

    async def _retrieve_and_compose(self, transcript: str, turn_id: str) -> list[dict[str, str]]:
        """POST /retrieve and return the LLMMessagesAppendFrame payload.

        Returns either the grounded messages (on in_scope=true + chunks)
        or the refusal messages (on in_scope=false, empty chunks, or
        non-2xx). Raises on transport / timeout so the caller's except
        handler logs it and falls back.
        """
        # tenant_id / session_id are checked non-None by is_enabled.
        assert self._tenant_id is not None
        assert self._session_id is not None

        payload = {
            "tenant_id": str(self._tenant_id),
            "session_id": str(self._session_id),
            "turn_id": turn_id,
            "transcript": transcript,
        }
        timeout = httpx.Timeout(self._timeout_ms / 1000, connect=1.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
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

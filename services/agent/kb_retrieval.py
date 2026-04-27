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

import secrets
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

# Default request timeout. Issue #49: lowered from 2000 ms to 500 ms so
# the constitution's p95 ≤ 800 ms end-to-end SLO has actual room for
# the LLM + TTS legs after retrieval. The retriever's own
# `rewriter_timeout_ms` (default 1500 ms) is now larger than this on
# purpose — the retriever fails-closed to refusal internally on
# rewriter timeout, returning a structured 200 well before the agent
# side gives up. If the retriever is itself slow (DB pause, container
# restart), the agent stops waiting at 500 ms and routes to refusal.
_DEFAULT_TIMEOUT_MS = 500


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
        internal_token: str = "",
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._retriever_url = retriever_url.rstrip("/")
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._internal_token = internal_token
        self._timeout_ms = timeout_ms
        # Issue #48: per-instance salt prefixed onto the monotonic
        # turn counter. Without it, an agent restart inside the same
        # session resets `_turn_counter = 0` and replays `t-00001`,
        # which collides with the already-written
        # UNIQUE(session_id, turn_id) row in retrieval_traces and
        # silently outage RAG for the rest of the session.
        # 4 hex chars = ~16-bit collision resistance per restart, which
        # is plenty for "two restarts in the same session" (vanishingly
        # rare).
        self._instance_salt = secrets.token_hex(2)
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
                "retriever_url_set={url_set} token_set={token_set}",
                reason=reason,
                tenant_set=tenant_id is not None,
                session_set=session_id is not None,
                url_set=bool(self._retriever_url),
                token_set=bool(self._internal_token),
            )
        else:
            # Misconfiguration alarm: enabled (tenant + session + url) but
            # token empty means every /retrieve will return 401 → refusal.
            # Silent in earlier versions because is_enabled doesn't gate
            # on the token (deliberate — a future deploy can run without
            # auth on a closed network). Surface it once at construction
            # so on-call sees the cause, not just the per-turn 401 noise.
            # See review findings sec-002 / adv-005 / correctness-005.
            if not self._internal_token:
                logger.warning(
                    "kb_retrieval.token_missing reason="
                    "retriever_url and tenant/session set but RETRIEVER_INTERNAL_TOKEN empty; "
                    "every /retrieve will return 401 and route to refusal"
                )
            headers = {"X-Internal-Token": self._internal_token} if self._internal_token else None
            # Issue #49 SLO is 800 ms p95 first-audio. Earlier this
            # carried `connect=1.0` while the read timeout was 0.5 s,
            # so DNS-slow / TCP-slow could spend the whole read budget
            # and another second on top — total 1.5 s, blowing past the
            # SLO. Bound connect by the same budget; a slow connect
            # routes to refusal at the same boundary as a slow read.
            timeout_s = self._timeout_ms / 1000
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s, connect=timeout_s),
                headers=headers,
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
        turn_id = f"{self._instance_salt}-{self._turn_counter:05d}"

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
            # Distinct log levels per failure class so on-call can
            # triage rotation drift (401) vs upstream outage (503) vs
            # other:
            #   401 — auth drift, almost always operator action
            #         (rotate / re-deploy with matching tokens).
            #   503 — retriever side down or DB/embedder/rewriter
            #         transient; refusal cascade is expected & self-
            #         healing once the dep recovers.
            #   other — surprising; logged at WARNING with status code.
            if resp.status_code == 401:
                logger.error(
                    "kb_retrieval.auth_failed turn={turn} reason="
                    "401 from /retrieve; check RETRIEVER_INTERNAL_TOKEN "
                    "matches between agent and retriever",
                    turn=turn_id,
                )
            elif resp.status_code == 503:
                logger.warning(
                    "kb_retrieval.dep_unavailable turn={turn} status=503",
                    turn=turn_id,
                )
            else:
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

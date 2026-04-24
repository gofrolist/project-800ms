"""POST /retrieve — the core Helper/Guide NPC RAG endpoint.

Flow:

  1. Pydantic parses + validates the request body.
  2. `resolve_tenant` rejects unknown / suspended tenants with 400
     `unknown_tenant` (no existence-oracle — suspended vs. missing
     return the same message; Principle IV).
  3. Explicit checks on `npc_id` and `language` raise
     `UnsupportedNpc` / `UnsupportedLanguage` (400). We do NOT use
     Pydantic `Literal[...]` for these — a `Literal` mismatch surfaces
     as HTTP 422 with the generic Pydantic error shape, which
     violates the OpenAPI contract (must be 400 + error envelope).
  4. Rewriter call → `{query, in_scope}`. Timeout / malformed output
     raise typed errors → 503 via the exception handler.
  5. If out-of-scope: write trace, return `chunks=[]`. (US2 adds the
     timing-parity pad; for US1 this branch is a stub.)
  6. If in-scope: embed → hybrid_search → write trace → return chunks.

Constitution Principle IV: the `tenant_id` from the request is used
as the scope for every downstream query; `resolve_tenant` confirms it
corresponds to an active row first.

Constitution Principle V / spec FR-021: `retrieval_traces` gets a row
for every in-scope and out-of-scope turn with all writable columns
populated. Upstream-error paths (rewriter timeout, DB unavailable) do
NOT currently write a trace — US5 will expand trace coverage to
failure branches.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from loguru import logger

from config import get_settings
from db import get_session
from embedder import encode
from errors import UnsupportedLanguage, UnsupportedNpc
from hybrid_search import hybrid_search
from rewriter import REWRITER_VERSION, rewrite_and_classify
from schemas import (
    FusionComponentsOut,
    RetrievedChunkOut,
    RetrieveRequest,
    RetrieveResponse,
    StageTimings,
)
from tenants import resolve_tenant
from traces import RetrievalTraceData, write_trace

router = APIRouter(tags=["retrieve"])

_SUPPORTED_NPC = "helper_guide"
_SUPPORTED_LANGUAGE = "ru"


def _ms_since(start: float) -> int:
    """Round to a non-negative integer millisecond count."""
    return max(0, int((time.perf_counter() - start) * 1000))


def _serialize_chunks_for_trace(
    chunks: list,
) -> list[dict]:
    """Project retrieved chunks into the forensics shape stored in the DB.

    Keeps the trace row compact: the full chunk content lives in
    `kb_chunks`; the trace only needs the ID + score components so
    operators can re-run the same hybrid search later.
    """
    return [
        {
            "chunk_id": c.id,
            "score": c.score,
            "fusion_components": {
                "semantic": c.fusion_components.semantic,
                "lexical": c.fusion_components.lexical,
            },
        }
        for c in chunks
    ]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    # Fast rejections on scope violations before touching the DB.
    if req.npc_id != _SUPPORTED_NPC:
        logger.info(
            "retrieve.unsupported_npc tenant={tenant} npc={npc}",
            tenant=req.tenant_id,
            npc=req.npc_id,
        )
        raise UnsupportedNpc()
    if req.language != _SUPPORTED_LANGUAGE:
        logger.info(
            "retrieve.unsupported_language tenant={tenant} lang={lang}",
            tenant=req.tenant_id,
            lang=req.language,
        )
        raise UnsupportedLanguage()

    settings = get_settings()
    stage_timings: dict[str, int] = {}

    async with get_session() as session:
        tenant = await resolve_tenant(session, req.tenant_id)

        # --- Rewrite -----------------------------------------------------
        rw_start = time.perf_counter()
        rewrite_result = await rewrite_and_classify(
            req.transcript,
            history=[h.model_dump() for h in req.history],
            model=settings.rewriter_model,
        )
        stage_timings["rewrite"] = _ms_since(rw_start)

        # --- Out-of-scope stub ------------------------------------------
        # US2 extends this branch with the timing-parity pad + refusal
        # plumbing. In US1 we just persist the trace and return an
        # empty chunks list so downstream agents can route to refusal.
        if not rewrite_result.in_scope:
            # `total` is defined as the SUM of tracked stages, not
            # wall-clock. Session acquire + tenant resolve are
            # deliberately excluded — the spec's invariant
            # `total == sum(parts) ± 1` means the response's `total`
            # is only the time attributable to the tracked phases.
            # Wall-clock request latency is observed at the caller.
            stage_timings["total"] = sum(stage_timings.values())
            trace_data = RetrievalTraceData(
                tenant_id=tenant.id,
                session_id=req.session_id,
                turn_id=req.turn_id,
                raw_transcript=req.transcript,
                rewritten_query=rewrite_result.query,
                in_scope=False,
                rewriter_version=REWRITER_VERSION,
                retrieved_chunks=[],
                stage_timings_ms=stage_timings,
            )
            trace_id = await write_trace(session, trace_data)
            return RetrieveResponse(
                rewritten_query=rewrite_result.query,
                in_scope=False,
                chunks=[],
                stage_timings_ms=StageTimings(**stage_timings),
                trace_id=trace_id,
            )

        # --- Embed --------------------------------------------------------
        emb_start = time.perf_counter()
        query_embedding = await encode(rewrite_result.query)
        stage_timings["embed"] = _ms_since(emb_start)

        # --- Hybrid search ------------------------------------------------
        sql_start = time.perf_counter()
        chunks = await hybrid_search(
            session,
            tenant.id,
            rewrite_result.query,
            query_embedding,
            top_k=req.top_k,
        )
        stage_timings["sql"] = _ms_since(sql_start)

        stage_timings["total"] = sum(stage_timings.values())

        # --- Trace write --------------------------------------------------
        trace_data = RetrievalTraceData(
            tenant_id=tenant.id,
            session_id=req.session_id,
            turn_id=req.turn_id,
            raw_transcript=req.transcript,
            rewritten_query=rewrite_result.query,
            in_scope=True,
            rewriter_version=REWRITER_VERSION,
            retrieved_chunks=_serialize_chunks_for_trace(chunks),
            stage_timings_ms=stage_timings,
        )
        trace_id = await write_trace(session, trace_data)

    return RetrieveResponse(
        rewritten_query=rewrite_result.query,
        in_scope=True,
        chunks=[
            RetrievedChunkOut(
                id=c.id,
                title=c.title,
                content=c.content,
                score=c.score,
                fusion_components=FusionComponentsOut(
                    semantic=c.fusion_components.semantic,
                    lexical=c.fusion_components.lexical,
                ),
                metadata=c.metadata,
            )
            for c in chunks
        ],
        stage_timings_ms=StageTimings(**stage_timings),
        trace_id=trace_id,
    )

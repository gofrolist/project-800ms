"""POST /retrieve — the core Helper/Guide NPC RAG endpoint.

Flow:

  1. Pydantic parses + validates the request body. Pydantic validation
     failures are translated to 400 ``invalid_request`` by the handler
     registered in `errors.py` (per OpenAPI contract; FastAPI's default
     422 shape is not part of the contract).
  2. Explicit checks on `npc_id` and `language` raise
     `UnsupportedNpc` / `UnsupportedLanguage` (400). We do NOT use
     Pydantic `Literal[...]` for these — a `Literal` mismatch would
     again surface through the Pydantic validation path, which is
     useful to keep short-circuiting around (it lets us return a
     specific error code and message per the contract).
  3. `resolve_tenant` rejects unknown / suspended tenants with 400
     `unknown_tenant` (no existence-oracle — suspended vs. missing
     return the same message; Principle IV).
  4. Rewriter call → `{query, in_scope}`. Timeout / malformed output
     raise typed errors → 503 via the exception handler.
  5. If out-of-scope: write trace, return `chunks=[]`.
  6. If in-scope: embed → hybrid_search → write trace → return chunks.
     Embedder `ValueError` is translated to `EmbedderUnavailable` and
     SQLAlchemy exceptions are translated to `DbUnavailable` so both
     reach the exception handler as a typed 503 + Error envelope.

Forensics invariant (FR-021, spec 002): *every* turn writes a
`retrieval_traces` row, including failure paths. The failure branches
populate `error_class` with the `ErrorCode.value` that surfaced to the
caller, and the row captures whatever stages completed before the
failure. Failure-trace writes are best-effort — if the DB itself is
the failure, the except-handler re-raises the original error rather
than shadowing it with a trace-write exception.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from config import get_settings
from db import get_session
from embedder import encode
from errors import (
    DbUnavailable,
    EmbedderUnavailable,
    RetrieverError,
    UnsupportedLanguage,
    UnsupportedNpc,
)
from hybrid_search import hybrid_search, p50_for, record_p50
from rewriter import REWRITER_VERSION, RewriterResult, rewrite_and_classify
from schemas import (
    FusionComponentsOut,
    RetrievedChunkOut,
    RetrieveRequest,
    RetrieveResponse,
    StageTimings,
)
from tenants import Tenant, resolve_tenant
from traces import RetrievalTraceData, write_trace

router = APIRouter(tags=["retrieve"])

_SUPPORTED_NPC = "helper_guide"
_SUPPORTED_LANGUAGE = "ru"


def _ms_since(start: float) -> int:
    """Round to a non-negative integer millisecond count."""
    return max(0, int((time.perf_counter() - start) * 1000))


def _serialize_chunks_for_trace(chunks: list) -> list[dict]:
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


async def _emit_trace(
    session,
    req: RetrieveRequest,
    tenant: Tenant,
    stage_timings: dict[str, int],
    rewrite_result: RewriterResult | None,
    chunks: list,
    *,
    error_class: str | None,
):
    """Write one `retrieval_traces` row. Used by success + failure paths.

    `error_class` is the code value (`ErrorCode.*.value`) when the
    request ultimately errored, or None on success. Stage-timings
    `total` is the sum of whatever stages ran before the exit — the
    invariant (total == sum ± 1) is preserved by always appending
    before the raise.
    """
    stage_timings_with_total = {
        **stage_timings,
        "total": sum(stage_timings.values()),
    }
    trace_data = RetrievalTraceData(
        tenant_id=tenant.id,
        session_id=req.session_id,
        turn_id=req.turn_id,
        raw_transcript=req.transcript,
        rewritten_query=rewrite_result.query if rewrite_result is not None else None,
        in_scope=rewrite_result.in_scope if rewrite_result is not None else None,
        rewriter_version=REWRITER_VERSION,
        retrieved_chunks=_serialize_chunks_for_trace(chunks),
        stage_timings_ms=stage_timings_with_total,
        error_class=error_class,
    )
    return await write_trace(session, trace_data)


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
    rewrite_result: RewriterResult | None = None
    chunks: list = []

    async with get_session() as session:
        # resolve_tenant can raise UnknownTenant → no trace row (no
        # valid tenant scope to write under, which would otherwise
        # require a "system tenant" row the schema doesn't have).
        tenant = await resolve_tenant(session, req.tenant_id)

        try:
            # --- Rewrite ----------------------------------------------
            rw_start = time.perf_counter()
            try:
                rewrite_result = await rewrite_and_classify(
                    req.transcript,
                    history=[h.model_dump() for h in req.history],
                    model=settings.rewriter_model,
                )
            finally:
                # Always record the attempted rewriter wall-time, even
                # on timeout / malformed-output — the invariant
                # `total == sum(parts)` and the forensics story both
                # need it populated.
                stage_timings["rewrite"] = _ms_since(rw_start)

            # --- Out-of-scope refusal branch (US2) ----------------------
            # Skip embed+sql but pad the wall-time so the round trip
            # mirrors the warm in-scope p50, preventing a side-channel
            # leak of scope classification (SC-008). The pad target is
            # 0 ms when this tenant has no recorded in-scope latency
            # yet — that's the correct degradation (a brand-new tenant
            # has no leak surface to measure against).
            if not rewrite_result.in_scope:
                pad_target_ms = p50_for(tenant.id)
                pad_start = time.perf_counter()
                if pad_target_ms > 0:
                    await asyncio.sleep(pad_target_ms / 1000.0)
                stage_timings["pad"] = _ms_since(pad_start)

                trace_id = await _emit_trace(
                    session,
                    req,
                    tenant,
                    stage_timings,
                    rewrite_result,
                    chunks=[],
                    error_class=None,
                )
                stage_timings_with_total = {
                    **stage_timings,
                    "total": sum(stage_timings.values()),
                }
                return RetrieveResponse(
                    rewritten_query=rewrite_result.query,
                    in_scope=False,
                    chunks=[],
                    stage_timings_ms=StageTimings(**stage_timings_with_total),
                    trace_id=trace_id,
                )

            # --- Embed -------------------------------------------------
            emb_start = time.perf_counter()
            try:
                query_embedding = await encode(rewrite_result.query)
            except ValueError as exc:
                # Shape mismatch or corrupt encoder state. Treat as
                # infra outage so the caller fails closed to refusal.
                logger.warning(
                    "retrieve.embedder_value_error message={message}",
                    message=str(exc),
                )
                raise EmbedderUnavailable() from exc
            finally:
                stage_timings["embed"] = _ms_since(emb_start)

            # --- Hybrid search -----------------------------------------
            sql_start = time.perf_counter()
            try:
                chunks = await hybrid_search(
                    session,
                    tenant.id,
                    rewrite_result.query,
                    query_embedding,
                    top_k=req.top_k,
                )
            except SQLAlchemyError as exc:
                # DB outage, pool timeout, schema drift. Exception
                # message may contain DSN fragments so we log only
                # the class name.
                logger.warning("retrieve.sql_error kind={kind}", kind=type(exc).__name__)
                raise DbUnavailable() from exc
            finally:
                stage_timings["sql"] = _ms_since(sql_start)

            # Record this in-scope round-trip's embed+sql cost so the
            # next refusal turn for this tenant can pad to its median.
            # Excludes rewrite (shared between branches) so the pad
            # mirrors only the work the refusal branch SKIPS.
            record_p50(tenant.id, stage_timings["embed"] + stage_timings["sql"])

            # --- Success trace + response ------------------------------
            trace_id = await _emit_trace(
                session,
                req,
                tenant,
                stage_timings,
                rewrite_result,
                chunks,
                error_class=None,
            )

        except RetrieverError as exc:
            # Write a failure trace row before letting the exception
            # reach the handler. `get_session()` rolls back on any
            # raised exception — we must commit the trace row
            # explicitly so it survives, then re-raise.
            #
            # Best-effort: if the DB itself is down (DbUnavailable),
            # the write or the commit may raise. Swallow that so we
            # don't shadow the original error.
            try:
                await _emit_trace(
                    session,
                    req,
                    tenant,
                    stage_timings,
                    rewrite_result,
                    chunks,
                    error_class=exc.code.value,
                )
                await session.commit()
            except Exception as trace_exc:  # noqa: BLE001
                logger.warning(
                    "retrieve.failure_trace_write_failed primary={primary} trace_kind={trace_kind}",
                    primary=exc.code.value,
                    trace_kind=type(trace_exc).__name__,
                )
            raise

    stage_timings_with_total = {
        **stage_timings,
        "total": sum(stage_timings.values()),
    }
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
        stage_timings_ms=StageTimings(**stage_timings_with_total),
        trace_id=trace_id,
    )

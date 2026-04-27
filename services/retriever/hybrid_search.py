"""Hybrid search — weighted-sum fusion of pgvector semantic + Russian tsvector lexical.

A single SQL statement with two CTEs (top-20 each) fused by weighted
sum: `0.7 * semantic + 0.3 * lexical`. Both CTEs already filter to one
tenant, so the final outer join cannot mix tenants. Returns the top-k
`RetrievedChunk` records ordered by fused score DESC.

Constitution Principle IV: every query layer carries `WHERE tenant_id =
:tid`. The test `TestTenantIsolation::test_no_cross_tenant_leakage`
(services/retriever/tests/test_hybrid_search.py) is the enforcing
boundary — removing any of the tenant_id predicates would make that
test fail by design.

Research R4 notes:
  * HNSW index parameters (m=16, ef_construction=200) live in migration
    0004_kb_chunks. ef_search is the default (40); we rely on that.
  * Fusion weights (0.7 / 0.3) are hard-coded here because any change
    requires a paired update in research.md (R2) and an eval re-run.
  * Semantic similarity is `1 - cosine_distance`. For normalized
    BGE-M3 vectors (the embedder normalizes by default) this lies in
    [0, 1]; the `<=>` operator returns cosine distance in the same range.
  * `plainto_tsquery('russian', ...)` handles empty / whitespace input
    by returning an empty tsquery that matches nothing — the lexical
    leg silently drops and fusion reduces to pure semantic, which is
    the right degradation for a noisy STT fragment.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_SEMANTIC_WEIGHT = 0.7
_LEXICAL_WEIGHT = 0.3
_CTE_LIMIT = 20

# ---------------------------------------------------------------------------
# Per-tenant latency stats — drives the US2 timing-parity pad (T038/T039).
# ---------------------------------------------------------------------------
#
# Constitution Principle IV: every observable side channel must respect
# tenant isolation. This means the rolling window MUST be keyed by
# tenant_id and MUST live in a namespace separate from
# `apps/api/rate_limit.py` (the xff-spoof learning: when two unrelated
# bucketing systems share the same dict, an unrelated regression in one
# can change the eviction behaviour of the other).
#
# Implementation:
#   * One deque per tenant, capped at `_LATENCY_WINDOW_MAX` samples
#     OR `_LATENCY_WINDOW_SECONDS`, whichever evicts first. For
#     tenants below ~3.3 RPS sustained the time bound dominates; for
#     busier tenants the count cap dominates and the effective
#     window shrinks proportionally with load. Pad still tracks
#     recent latency under both regimes — only the documented
#     "rolling 60-s" window is approximate (review finding perf-003).
#   * Each sample is `(timestamp_seconds, latency_ms)`. The window is
#     trimmed to the most recent `_LATENCY_WINDOW_SECONDS` on every
#     read so a quiet tenant doesn't return stale data.
#   * `p50_for(tenant_id)` returns 0 on an empty / too-stale window —
#     which is the correct degradation: a brand-new tenant with no
#     in-scope traffic gets pad=0 ms (matching the equally-fast
#     rewrite-only outcome).
#   * The state is process-local. Multiple retriever replicas each
#     compute their own p50; we accept the slight per-replica variance
#     for the simplicity of not standing up a Redis dependency.

_LATENCY_WINDOW_SECONDS = 60.0
# Hard cap per tenant — bounds memory under load. Caps the effective
# window at 200/QPS seconds; for sustained QPS > 3.3 the count cap
# wins and the window shrinks below the nominal 60s.
_LATENCY_WINDOW_MAX = 200

_latency_windows: dict[uuid.UUID, deque[tuple[float, int]]] = {}


def _reset_latency_stats() -> None:
    """Wipe every tenant's latency window — used by tests so a prior
    test's recorded p50 doesn't bleed into the next one."""
    _latency_windows.clear()


def record_p50(tenant_id: uuid.UUID, latency_ms: int) -> None:
    """Record one in-scope latency sample for a tenant.

    Caller (the in-scope success branch in retrieve.py) passes the
    embed+sql wall-time so the refusal pad can mirror the cost the
    rewriter-only branch SKIPS. Rewrite is shared between branches
    and so excluded.

    Negative or zero latencies are ignored — a 0-ms sample would
    unconditionally pull p50 toward 0 and defeat the purpose.
    """
    if latency_ms <= 0:
        return
    window = _latency_windows.setdefault(tenant_id, deque(maxlen=_LATENCY_WINDOW_MAX))
    window.append((time.monotonic(), latency_ms))


def p50_for(tenant_id: uuid.UUID) -> int:
    """Return the median in-scope latency for `tenant_id` over the
    rolling 60-s window. Returns 0 when no qualifying samples exist.

    Stale samples (older than `_LATENCY_WINDOW_SECONDS`) are dropped
    eagerly here so a tenant that had a burst and went quiet doesn't
    keep padding refusals against minute-old data.
    """
    window = _latency_windows.get(tenant_id)
    if not window:
        return 0
    cutoff = time.monotonic() - _LATENCY_WINDOW_SECONDS
    while window and window[0][0] < cutoff:
        window.popleft()
    if not window:
        return 0
    return int(statistics.median(latency for _, latency in window))


@dataclass(frozen=True)
class FusionComponents:
    """Exposed so callers can reproduce the fusion math or log components."""

    semantic: float
    lexical: float


@dataclass(frozen=True)
class RetrievedChunk:
    """Single hybrid-search result. Matches the `RetrievedChunk` schema
    in `specs/002-helper-guide-npc/contracts/retrieve.openapi.yaml`.
    """

    id: int
    title: str
    content: str
    score: float
    fusion_components: FusionComponents
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
#
# Two CTEs → full outer join → weighted sum → top-k. The outer `JOIN
# kb_chunks kc ON kc.id = fused.id AND kc.tenant_id = :tid` is belt-and-
# -braces tenant isolation — the CTEs already enforce it, but the extra
# predicate means a future refactor that removes one CTE filter still
# leaves the tenant guard in place at the outer layer.
_HYBRID_SQL = text(
    """
    WITH sem AS (
        SELECT id,
               1 - (embedding <=> CAST(:emb AS vector)) AS semantic_score
        FROM kb_chunks
        WHERE tenant_id = :tid
        ORDER BY embedding <=> CAST(:emb AS vector)
        LIMIT :cte_limit
    ),
    lex AS (
        SELECT id,
               ts_rank_cd(content_tsv, plainto_tsquery('russian', :q))
                   AS lexical_score
        FROM kb_chunks
        WHERE tenant_id = :tid
          AND content_tsv @@ plainto_tsquery('russian', :q)
        ORDER BY lexical_score DESC
        LIMIT :cte_limit
    ),
    fused AS (
        SELECT
            COALESCE(sem.id, lex.id) AS id,
            COALESCE(sem.semantic_score, 0)::float AS sem_score,
            COALESCE(lex.lexical_score, 0)::float AS lex_score,
            :w_sem * COALESCE(sem.semantic_score, 0)::float
              + :w_lex * COALESCE(lex.lexical_score, 0)::float
                AS fused_score
        FROM sem
        FULL OUTER JOIN lex ON sem.id = lex.id
    )
    SELECT kc.id, kc.title, kc.content, kc.metadata,
           fused.sem_score, fused.lex_score, fused.fused_score
    FROM fused
    JOIN kb_chunks kc ON kc.id = fused.id AND kc.tenant_id = :tid
    ORDER BY fused.fused_score DESC
    LIMIT :k
    """
)


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    """Serialize a Python float list as a pgvector text literal.

    pgvector accepts `[1.0, 2.0, 3.0]` as a cast-able text form. A text
    literal avoids having to register a per-connection type adapter on
    the pool, which is fiddly to do reliably across async sessions and
    buys us nothing for a single-call-per-retrieval workload.
    """
    return "[" + ",".join(f"{x:.10g}" for x in embedding) + "]"


async def hybrid_search(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Run the two-CTE hybrid search for one tenant.

    Args:
        session: Async session; caller owns its lifecycle + rollback.
        tenant_id: MANDATORY. Every CTE plus the outer select filters
            on this predicate. A cross-tenant leak here is a CRITICAL
            constitution Principle IV violation.
        query_text: The rewriter's output (or the raw transcript if the
            rewriter failed closed). Passed to `plainto_tsquery` — an
            empty/whitespace query silently drops the lexical leg.
        query_embedding: 1024-dim dense vector from BGE-M3, normalized.
        top_k: Post-fusion cap on returned chunks.

    Returns:
        List of `RetrievedChunk` ordered by fused score DESC. Empty if
        the tenant has no chunks, neither CTE matched, or `top_k <= 0`.
    """
    result = await session.execute(
        _HYBRID_SQL,
        {
            "emb": _embedding_to_vector_literal(query_embedding),
            "tid": tenant_id,
            "q": query_text,
            "k": top_k,
            "cte_limit": _CTE_LIMIT,
            "w_sem": _SEMANTIC_WEIGHT,
            "w_lex": _LEXICAL_WEIGHT,
        },
    )

    chunks: list[RetrievedChunk] = []
    for row in result.mappings():
        chunks.append(
            RetrievedChunk(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                score=float(row["fused_score"]),
                fusion_components=FusionComponents(
                    semantic=float(row["sem_score"]),
                    lexical=float(row["lex_score"]),
                ),
                metadata=dict(row["metadata"]) if row["metadata"] else {},
            )
        )
    return chunks

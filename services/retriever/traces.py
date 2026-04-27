"""Trace writer — INSERT-only access to `retrieval_traces`.

One public function, `write_trace(session, data) -> UUID`. Validates
the stage-timings bookkeeping invariant BEFORE insert — a malformed
trace never reaches disk. Violation is always a programmer error
upstream; surfacing it as `ValueError` here prevents silent divergence
between the response body's `stage_timings_ms.total` and the row we
later use for forensics (spec 002 FR-021).

`retrieval_traces` is append-only. This module contains no UPDATE path
by design; US5 / T059 will add an assertion guarding that invariant.

Caller owns the session lifecycle + commit/rollback. We issue
`session.flush()` here only to populate the server-generated `id` so
the /retrieve response can include `trace_id` synchronously with the
row write.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from models import RetrievalTrace

# Rounding tolerance when reconciling `total` against sum-of-parts.
# Individual stages are measured in float seconds and rounded to int
# ms independently; the total is stamped at the end of the request.
# One-ms slack absorbs double-rounding. Anything larger is a real bug.
_TOTAL_TOLERANCE_MS = 1


@dataclass(frozen=True)
class RetrievalTraceData:
    """Pure-Python staging of a row about to be INSERTed.

    Separated from the SQLAlchemy mapped class so tests can build these
    cheaply and the invariant check operates on immutable values. The
    ORM class is a thin persistence adapter on top of this shape.
    """

    tenant_id: uuid.UUID
    session_id: uuid.UUID
    turn_id: str
    raw_transcript: str
    rewritten_query: str | None
    in_scope: bool | None
    rewriter_version: str
    retrieved_chunks: list[dict[str, Any]]
    stage_timings_ms: dict[str, int]
    final_reply_text: str | None = None
    error_class: str | None = None


def _validate_stage_timings(stage_timings_ms: dict[str, int]) -> None:
    """Enforce `total == sum(non-total keys) ± 1 ms`.

    The non-negotiable guarantee for operators reconstructing a turn:
    the published `stage_timings_ms.total` must account for every
    recorded stage. If not, the trace row and the HTTP response would
    disagree about wall time and the forensics path becomes unreliable.
    """
    if "total" not in stage_timings_ms:
        raise ValueError(
            f"stage_timings_ms must include 'total'; got keys {sorted(stage_timings_ms.keys())}"
        )
    total = stage_timings_ms["total"]
    parts = {k: v for k, v in stage_timings_ms.items() if k != "total"}
    parts_sum = sum(parts.values())
    if abs(total - parts_sum) > _TOTAL_TOLERANCE_MS:
        raise ValueError(
            f"stage_timings_ms.total={total} disagrees with "
            f"sum(parts)={parts_sum} beyond ±{_TOTAL_TOLERANCE_MS} ms "
            f"(stages={sorted(parts.keys())})"
        )


async def write_trace(session: AsyncSession, data: RetrievalTraceData) -> uuid.UUID:
    """INSERT `data` into retrieval_traces and return the row's UUID.

    Args:
        session: Active async session; caller owns commit/rollback.
        data: Fully-populated trace payload. `stage_timings_ms` must
            satisfy the total=sum invariant.

    Returns:
        Server-assigned UUID (via `gen_random_uuid()` default). Exposed
        to the HTTP layer as `trace_id` in the response envelope.

    Raises:
        ValueError: If `stage_timings_ms` fails the bookkeeping check.
    """
    _validate_stage_timings(data.stage_timings_ms)

    row = RetrievalTrace(
        tenant_id=data.tenant_id,
        session_id=data.session_id,
        turn_id=data.turn_id,
        raw_transcript=data.raw_transcript,
        rewritten_query=data.rewritten_query,
        in_scope=data.in_scope,
        rewriter_version=data.rewriter_version,
        retrieved_chunks=data.retrieved_chunks,
        stage_timings_ms=data.stage_timings_ms,
        final_reply_text=data.final_reply_text,
        error_class=data.error_class,
    )
    session.add(row)
    await session.flush()  # populates row.id from the server default
    return row.id

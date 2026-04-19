"""GET /v1/usage — per-tenant daily aggregate.

Rolls up the sessions table so a tenant can self-serve "how much audio
did we consume this month". audio_seconds is populated by the LiveKit
webhook on room_finished, so a tenant with lots of `pending` sessions
(never dispatched to the agent) or `active` sessions (still in flight)
will see lower numbers until those terminate.

The billing surface is deliberately read-only and tenant-scoped —
aggregations span exactly one tenant per call. Admin API (Phase 2d)
will add a cross-tenant variant.
"""

from __future__ import annotations

import datetime
import uuid

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import TenantIdentity
from db import get_db
from errors import APIError
from models import Session as SessionRow
from rate_limit import enforce_tenant_rate_limit

router = APIRouter(prefix="/v1", tags=["usage"])

# Hard cap on the window callers can request. Aggregates over >1 year
# of rows are almost certainly a misuse — wire a paginated / async
# report if it turns out real customers need it.
_MAX_WINDOW_DAYS = 366


class DailyUsage(BaseModel):
    day: str  # ISO date
    sessions_count: int
    audio_seconds: int


class UsageResponse(BaseModel):
    from_date: str
    to_date: str
    total_sessions: int
    total_audio_seconds: int
    days: list[DailyUsage]


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Daily usage aggregate for the current tenant",
    description=(
        "Returns one row per UTC day in `[from, to)`. `audio_seconds` is "
        "the sum of `sessions.audio_seconds` for sessions whose "
        "`started_at` falls in the bucket — populated by the LiveKit "
        "`room_finished` webhook. Active / pending sessions contribute "
        "`sessions_count` but not `audio_seconds` until they terminate."
    ),
)
async def get_usage(
    request: Request,
    from_: datetime.date = Query(
        ...,
        alias="from",
        description="Inclusive start date (UTC), YYYY-MM-DD.",
    ),
    to: datetime.date = Query(
        ...,
        description="Exclusive end date (UTC), YYYY-MM-DD.",
    ),
    identity: TenantIdentity = Depends(enforce_tenant_rate_limit),
    db: AsyncSession = Depends(get_db),
) -> UsageResponse:
    if from_ >= to:
        raise APIError(422, "validation_error", "`from` must be strictly before `to`")
    if (to - from_).days > _MAX_WINDOW_DAYS:
        raise APIError(
            422,
            "validation_error",
            f"Window too large: max {_MAX_WINDOW_DAYS} days",
        )

    # Convert date → tz-aware datetime at UTC midnight so the comparison
    # against TIMESTAMPTZ columns works without server-side casts.
    start = datetime.datetime.combine(from_, datetime.time.min, tzinfo=datetime.UTC)
    end = datetime.datetime.combine(to, datetime.time.min, tzinfo=datetime.UTC)

    day_col = func.date_trunc("day", SessionRow.started_at).label("day")
    stmt = (
        select(
            day_col,
            func.count(SessionRow.id).label("sessions_count"),
            func.coalesce(func.sum(SessionRow.audio_seconds), 0).label("audio_seconds"),
        )
        .where(SessionRow.tenant_id == uuid.UUID(identity.tenant_id))
        .where(SessionRow.started_at >= start)
        .where(SessionRow.started_at < end)
        .group_by(day_col)
        .order_by(day_col)
    )
    rows = (await db.execute(stmt)).all()

    days: list[DailyUsage] = []
    total_sessions = 0
    total_audio = 0
    for row in rows:
        day_dt = row.day
        total_sessions += row.sessions_count
        total_audio += int(row.audio_seconds)
        days.append(
            DailyUsage(
                day=day_dt.date().isoformat(),
                sessions_count=row.sessions_count,
                audio_seconds=int(row.audio_seconds),
            )
        )

    return UsageResponse(
        from_date=from_.isoformat(),
        to_date=to.isoformat(),
        total_sessions=total_sessions,
        total_audio_seconds=total_audio,
        days=days,
    )

"""Structured logging + Prometheus metrics.

Two concerns that fit in one module because they share the same
context plumbing:

    tenant_id_var / room_var — ContextVars populated by middleware and
    auth deps; read by the JSON formatter and the Prometheus label
    extractor.

Deliberately sticks to the stdlib `logging` module — no structlog
dependency — because FastAPI / uvicorn / starlette already funnel
through it and we want their records to pick up our context too.

Prometheus metrics are opt-in: `install_metrics(app)` mounts `/metrics`
and adds a middleware that observes every request. Scrape from the
internal network; there's no auth on /metrics.
"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from request_id import request_id_var

# Sentinel for "no tenant yet" — cleaner to spot than an empty string.
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")
tenant_slug_var: ContextVar[str] = ContextVar("tenant_slug", default="-")
room_var: ContextVar[str] = ContextVar("room", default="-")


# ─── Structured JSON log formatter ────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """One JSON line per log record.

    Captures the request_id / tenant_id / tenant_slug / room context
    vars on top of the usual log fields. Exception info collapses onto
    an `exc_info` string field — no multi-line records, so log pipelines
    don't need to do record reassembly.
    """

    # Fields built into LogRecord that we don't want to propagate as JSON
    # keys (they're either redundant, huge, or just internal).
    _SKIP = {
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "name",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "tenant_id": tenant_id_var.get(),
            "tenant_slug": tenant_slug_var.get(),
            "room": room_var.get(),
        }
        # Anything non-standard that the caller passed as `extra=`.
        for k, v in record.__dict__.items():
            if k not in self._SKIP and k not in payload:
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_json_logging(level: str | int = "INFO") -> None:
    """Install the JSON formatter on the root handler.

    Called once at startup. Idempotent — resets existing handlers so a
    reload (e.g. during a pytest run that re-imports main) doesn't stack
    formatters and emit duplicate lines.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn / fastapi add their own handlers during app boot; rein them
    # in so their records also go through our JSON formatter.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(noisy)
        lg.handlers.clear()
        lg.propagate = True


# ─── Prometheus metrics ───────────────────────────────────────────────────


# Labels: method, path (template, not full URL), status (str), tenant_slug.
# path uses the route template ("/v1/sessions/{room}") not the literal
# URL so cardinality stays bounded — otherwise one tenant probing random
# rooms blows up metric-series count.
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ("method", "path", "status", "tenant_slug"),
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ("method", "path", "tenant_slug"),
    # Tuned for an 800ms-budget voice API: most requests are auth + DB
    # round-trips and should land well under 200ms. Buckets skew to the
    # fast end accordingly.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# LiveKit server-side API errors swallowed by ``_delete_livekit_room``.
# Incremented every time the best-effort cleanup on DELETE /v1/sessions/{room}
# sees an exception from the LiveKit SDK. The DB-side close still commits,
# so these failures are invisible in HTTP metrics (the response is 200) —
# this counter is the only signal a prolonged LiveKit outage is producing
# zombie rooms. Alert: ``rate(api_livekit_delete_failures_total[5m]) > 0``
# sustained over 10 minutes.
#
# Labeled by exception class so distinct failure modes (connect error vs
# timeout vs OS error) are separable on the dashboard.
livekit_delete_failures_total = Counter(
    "api_livekit_delete_failures_total",
    "Count of LiveKit delete_room calls that failed and were swallowed by _delete_livekit_room.",
    ("exception_type",),
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Observes every request into the counter + histogram."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip /metrics itself to avoid recursive measurement of Prometheus
        # scrapes (those are bots, not interesting business traffic).
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response: Response | None = None
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            path = _route_template(request) or request.url.path
            tenant = tenant_slug_var.get()
            http_requests_total.labels(
                method=request.method,
                path=path,
                status=status,
                tenant_slug=tenant,
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                path=path,
                tenant_slug=tenant,
            ).observe(elapsed)


def _route_template(request: Request) -> str | None:
    """Extract the matched route template (e.g. /v1/sessions/{room}).

    Starlette doesn't fill `request.scope['route']` until after routing
    runs, and for unmatched paths returns nothing — callers fall back to
    the literal URL in that case.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else None


async def metrics_endpoint() -> Response:
    """Plain-text Prometheus exposition format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def install_metrics(app: FastAPI) -> None:
    """Mount the middleware + /metrics endpoint on an app instance."""
    app.add_middleware(PrometheusMiddleware)
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )

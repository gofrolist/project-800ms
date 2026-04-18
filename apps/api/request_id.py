"""Request ID middleware — mints a ULID per request, exposes it via
contextvar + response header so every log line can reference it.

Why ULID instead of UUID: time-ordered, URL-safe (Crockford base32),
26 chars instead of 36. Trivially sortable when debugging.

Why contextvar instead of threading through every function: loguru's
sink reads `request_id_var.get()` directly — zero boilerplate at call
sites. Works across async awaits (contextvars survive `await`).
"""

from __future__ import annotations

from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from ulid import ULID

# "-" sentinel is what log lines show when outside a request context
# (e.g. startup logs). Easier to spot than an empty string.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Binds a ULID to request_id_var for the lifetime of the request.

    Honors an inbound X-Request-ID when it looks like a valid ULID; this
    lets game backends propagate their own trace IDs end-to-end. Otherwise
    generate a fresh one.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        inbound = request.headers.get("X-Request-ID", "")
        rid = inbound if _looks_like_ulid(inbound) else str(ULID())
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response


def _looks_like_ulid(s: str) -> bool:
    # ULIDs are 26 chars of Crockford base32. Don't bother with strict
    # parsing — any short alphanumeric-ish ID is fine to echo.
    return 8 <= len(s) <= 64 and s.replace("-", "").replace("_", "").isalnum()


def current_request_id() -> str:
    """Return the active request ID, or '-' if no request is in scope."""
    return request_id_var.get()

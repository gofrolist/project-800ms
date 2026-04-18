"""Unified error envelope for /v1/* endpoints.

Every non-2xx response on /v1/* is shaped like:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

The `code` is a stable machine-readable slug; `message` is human-readable
and safe to surface to end users. `request_id` echoes the middleware-issued
ULID so support tickets can correlate.

Legacy endpoints (health, and the about-to-be-deleted /sessions) keep their
existing shapes to avoid breaking in-flight deployments.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from request_id import current_request_id

# Canonical error codes. Anything returned from /v1/* should use one of
# these; otherwise we end up with a soup of ad-hoc strings.
ERROR_CODES: set[str] = {
    "unauthenticated",
    "forbidden",
    "not_found",
    "conflict",
    "validation_error",
    "rate_limited",
    "payload_too_large",
    "agent_unavailable",
    "internal_error",
}


class APIError(HTTPException):
    """Raise to produce a {error: {...}} envelope response.

    Usage:
        raise APIError(401, "unauthenticated", "Missing X-API-Key header")
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if code not in ERROR_CODES:
            # Fail loudly in dev; in prod the global handler still produces
            # a valid envelope but with `internal_error` code.
            raise ValueError(f"Unknown error code: {code}")
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.extra = extra or {}


def _envelope(code: str, message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": current_request_id(),
        }
    }
    if extra:
        body["error"].update(extra)
    return body


async def api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.detail, exc.extra),
    )


async def http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    # Best-effort mapping from status code to our canonical codes. Legacy
    # callers that `raise HTTPException(404)` still get a valid envelope.
    status_code = exc.status_code
    code = {
        status.HTTP_401_UNAUTHORIZED: "unauthenticated",
        status.HTTP_403_FORBIDDEN: "forbidden",
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_409_CONFLICT: "conflict",
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: "payload_too_large",
        status.HTTP_422_UNPROCESSABLE_ENTITY: "validation_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
        status.HTTP_503_SERVICE_UNAVAILABLE: "agent_unavailable",
    }.get(status_code, "internal_error")
    message = (
        exc.detail if isinstance(exc.detail, str) else "An error occurred processing the request"
    )
    return JSONResponse(status_code=status_code, content=_envelope(code, message))


async def validation_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Surface the pydantic errors array under `extra.errors` — machine-
    # actionable for clients that want field-level feedback.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_envelope(
            "validation_error",
            "Request body failed validation",
            extra={"errors": exc.errors()},
        ),
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Swallow the internals — don't leak stack traces. The logger records the
    # traceback so ops can correlate via request_id.
    import logging

    logging.getLogger("project-800ms.api").exception("Unhandled exception on request: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope("internal_error", "Internal server error"),
    )


def install(app: FastAPI) -> None:
    """Register all envelope-producing exception handlers on the app."""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

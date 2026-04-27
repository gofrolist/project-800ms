"""Typed error classes + FastAPI exception handler.

Error codes match the `Error.error` enum in
`specs/002-helper-guide-npc/contracts/retrieve.openapi.yaml`. Messages
are PII-safe — no KB content, no raw transcript, no internal prompts
(constitution Principle V; spec 002 FR-022).
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    """Machine-readable error codes."""

    INVALID_REQUEST = "invalid_request"
    UNKNOWN_TENANT = "unknown_tenant"
    UNSUPPORTED_NPC = "unsupported_npc"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    REWRITER_TIMEOUT = "rewriter_timeout"
    REWRITER_MALFORMED_OUTPUT = "rewriter_malformed_output"
    EMBEDDER_UNAVAILABLE = "embedder_unavailable"
    DB_UNAVAILABLE = "db_unavailable"
    INTERNAL_ERROR = "internal_error"


# Map each error code to its HTTP status. Kept explicit so a future
# reviewer can audit "which errors surface as 400 vs 503" at a glance.
_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNKNOWN_TENANT: 400,
    ErrorCode.UNSUPPORTED_NPC: 400,
    ErrorCode.UNSUPPORTED_LANGUAGE: 400,
    ErrorCode.REWRITER_TIMEOUT: 503,
    ErrorCode.REWRITER_MALFORMED_OUTPUT: 503,
    ErrorCode.EMBEDDER_UNAVAILABLE: 503,
    ErrorCode.DB_UNAVAILABLE: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class RetrieverError(Exception):
    """Base for all typed errors raised by the retriever.

    Carries a machine-readable code, a short human message (PII-safe),
    and optionally a `trace_id` so the agent can correlate the failure
    with a `retrieval_traces` row already on disk.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        trace_id: UUID | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.trace_id = trace_id

    @property
    def http_status(self) -> int:
        return _STATUS_BY_CODE[self.code]

    def to_envelope(self) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "error": self.code.value,
            "message": self.message,
        }
        if self.trace_id is not None:
            envelope["trace_id"] = str(self.trace_id)
        return envelope


# Convenience subclasses so call sites read cleanly:
#   raise InvalidRequest("transcript must be non-empty")


class InvalidRequest(RetrieverError):
    def __init__(self, message: str, *, trace_id: UUID | None = None) -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, message, trace_id=trace_id)


class UnknownTenant(RetrieverError):
    def __init__(self, message: str = "unknown tenant", *, trace_id: UUID | None = None) -> None:
        super().__init__(ErrorCode.UNKNOWN_TENANT, message, trace_id=trace_id)


class UnsupportedNpc(RetrieverError):
    def __init__(
        self, message: str = "unsupported npc_id", *, trace_id: UUID | None = None
    ) -> None:
        super().__init__(ErrorCode.UNSUPPORTED_NPC, message, trace_id=trace_id)


class UnsupportedLanguage(RetrieverError):
    def __init__(
        self, message: str = "unsupported language", *, trace_id: UUID | None = None
    ) -> None:
        super().__init__(ErrorCode.UNSUPPORTED_LANGUAGE, message, trace_id=trace_id)


class RewriterTimeout(RetrieverError):
    def __init__(
        self, message: str = "rewriter timed out", *, trace_id: UUID | None = None
    ) -> None:
        super().__init__(ErrorCode.REWRITER_TIMEOUT, message, trace_id=trace_id)


class RewriterMalformedOutput(RetrieverError):
    def __init__(
        self,
        message: str = "rewriter returned malformed output",
        *,
        trace_id: UUID | None = None,
    ) -> None:
        super().__init__(ErrorCode.REWRITER_MALFORMED_OUTPUT, message, trace_id=trace_id)


class EmbedderUnavailable(RetrieverError):
    def __init__(
        self, message: str = "embedder unavailable", *, trace_id: UUID | None = None
    ) -> None:
        super().__init__(ErrorCode.EMBEDDER_UNAVAILABLE, message, trace_id=trace_id)


class DbUnavailable(RetrieverError):
    def __init__(self, message: str = "db unavailable", *, trace_id: UUID | None = None) -> None:
        super().__init__(ErrorCode.DB_UNAVAILABLE, message, trace_id=trace_id)


class InternalError(RetrieverError):
    def __init__(self, message: str = "internal error", *, trace_id: UUID | None = None) -> None:
        super().__init__(ErrorCode.INTERNAL_ERROR, message, trace_id=trace_id)


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the RetrieverError → Error-envelope JSON handler onto the app.

    Call once from `create_app()`. Any uncaught `RetrieverError` raised by
    route handlers surfaces as the envelope shape documented in
    `contracts/retrieve.openapi.yaml`, with the correct HTTP status.
    """

    @app.exception_handler(RetrieverError)
    async def _handle_retriever_error(_request: Request, exc: RetrieverError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope())

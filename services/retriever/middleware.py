"""HTTP middleware for the retriever service.

Currently exports one middleware:

* ``BodySizeLimitMiddleware`` — rejects requests whose declared
  ``Content-Length`` exceeds a configured cap, BEFORE uvicorn buffers
  the body into memory. Closes issue #56: the auth dep runs at the
  route layer, after the body has already been read off the wire, so
  a co-resident attacker holding the bearer secret could still flood
  /retrieve with multi-MB bodies and OOM the container.

Pydantic ``max_length`` field constraints (transcript ≤ 4000 chars)
fire AFTER the body has been parsed; they do NOT gate the upstream
buffering. This middleware is the upstream gate.
"""

from __future__ import annotations

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp


# 64 KiB is comfortably above the realistic /retrieve request size
# (UUID ids + 6×4000-char history + 4000-char transcript) which tops
# out around 30 KiB. Real callers never approach this; an attacker
# trying to exhaust memory does.
DEFAULT_MAX_BODY_BYTES = 64 * 1024


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests larger than ``max_body_bytes`` on the
    ``Content-Length`` header. Returns ``413 payload_too_large`` in
    the flat error envelope.

    Requests without ``Content-Length`` (chunked transfer encoding)
    are passed through — uvicorn enforces its own per-connection
    streaming cap that bounds the worst case to a few MB. The vast
    majority of legitimate JSON requests carry an explicit length
    header.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > self._max_body_bytes:
                logger.warning(
                    "retriever.body_too_large length={length} cap={cap} path={path}",
                    length=length,
                    cap=self._max_body_bytes,
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "payload_too_large",
                            "message": f"Request body exceeds {self._max_body_bytes} bytes",
                        }
                    },
                )
        return await call_next(request)

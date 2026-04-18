"""project-800ms API.

Endpoints:
    GET  /health        — liveness
    POST /v1/sessions   — mint a LiveKit caller token (X-API-Key auth)
    GET  /v1/sessions/{room} — fetch session details

The legacy unauthenticated POST /sessions has been removed — every caller
must now authenticate via X-API-Key. See `routes/sessions.py`.
"""

from __future__ import annotations

import logging
import os

import errors
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rate_limit import limiter
from request_id import RequestIdMiddleware
from routes.sessions import router as sessions_router
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from settings import settings

logger = logging.getLogger("project-800ms.api")

_is_production = os.getenv("ENV", "").lower() == "production"

app = FastAPI(
    title="project-800ms API",
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

# Request ID must run before anything else so every downstream log +
# error response has an ID to reference.
app.add_middleware(RequestIdMiddleware)

# Register error handlers — every HTTP error on the app now produces the
# unified {error: {...}} envelope.
errors.install(app)


# ─── Rate limiting ────────────────────────────────────────────────────────
# One limiter, tenant-aware. `tenant_or_ip` keys authenticated requests by
# tenant_id (populated by get_current_tenant into request.state) and falls
# back to IP for unauthenticated routes like /health. slowapi's default
# limits are empty — enforcement is opt-in per route via @limiter.limit.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ─── Security headers ─────────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add baseline security headers to every response.

    The API only returns JSON, so most browser-targeted headers are belt-
    and-braces. They cost nothing and harden against content-type sniffing,
    clickjacking, and referrer leaks if the API ever serves HTML by mistake.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self' wss:; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'",
        )
        response.headers.setdefault(
            "Permissions-Policy",
            "microphone=(self), camera=(), geolocation=()",
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ─── CORS ─────────────────────────────────────────────────────────────────
# CORS origins are env-driven (CORS_ALLOWED_ORIGINS). Default is "*" for
# local dev; set explicit origins in prod. Per-tenant allowed_origins
# (stored in the tenants table) is the real access-control boundary — this
# global list is a belt-and-braces second layer (see Phase 1g).
if settings.cors_allowed_origins == ["*"]:
    logger.warning(
        "CORS is configured with wildcard origin (*). This is fine for local "
        "dev but MUST NOT be used in production. Set CORS_ALLOWED_ORIGINS to "
        "an explicit comma-separated list of origins."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["GET", "POST"],
    # Client sends Content-Type + X-API-Key + optional X-Request-ID.
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


# ─── Routes ───────────────────────────────────────────────────────────────
app.include_router(sessions_router)


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}

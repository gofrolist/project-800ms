"""project-800ms API.

Endpoints:
    GET  /health      — liveness
    POST /sessions    — mint a LiveKit caller token for the demo room (DEPRECATED)
    (new /v1/* endpoints land in subsequent commits)
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid

import errors
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from livekit import api as lkapi
from pydantic import BaseModel
from request_id import RequestIdMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
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
# unified {error: {...}} envelope. Does NOT affect legacy /sessions
# success responses, only the error paths.
errors.install(app)


# ─── Rate limiting ────────────────────────────────────────────────────────
# Per-remote-IP limits. /sessions is the high-value endpoint — each call
# mints a LiveKit token good for `session_ttl_seconds`, so a flood would
# saturate the LiveKit room and burn GPU compute on the agent. /health is
# rate-limited too (loosely) to prevent availability scanning abuse.
#
# Note: when behind a reverse proxy, configure proxy-forwarded headers and
# swap `get_remote_address` for a key function that reads X-Forwarded-For.
def _real_ip(request: Request) -> str:
    """Extract the real client IP when behind a trusted reverse proxy (Caddy).

    Only trusts X-Forwarded-For when the immediate peer is on the Docker
    bridge network (172.x.x.x), preventing header spoofing from direct clients.
    """
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    if xff and client_host.startswith("172."):
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_real_ip, default_limits=[])
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
# local dev; set explicit origins in prod or any deployment that issues real
# LiveKit credentials, since /sessions is unauthenticated and any allowed
# origin can mint caller tokens.
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
    # Only the headers the API actually receives. `*` is unnecessarily broad.
    allow_headers=["Content-Type"],
)


class SessionResponse(BaseModel):
    """What the browser needs to join a call."""

    url: str  # LiveKit ws URL (public)
    token: str  # JWT for this caller
    room: str  # Room name the agent is sitting in
    identity: str  # Caller identity embedded in the token


# Both handlers are sync (`def`, not `async def`) because they do no I/O —
# /health returns a constant and /sessions only signs a JWT (CPU-only HMAC).
# FastAPI runs sync handlers in a threadpool, which is correct here. Convert
# to `async def` if either gains real I/O (DB lookups, HTTP calls).


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionResponse)
@limiter.limit("5/minute")
async def create_session(request: Request) -> SessionResponse:
    """Create a private room and dispatch an agent into it."""
    room = f"room-{uuid.uuid4().hex[:8]}"
    identity = f"user-{uuid.uuid4().hex[:8]}"
    token = (
        lkapi.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(
            lkapi.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(datetime.timedelta(seconds=settings.session_ttl_seconds))
        .to_jwt()
    )

    # Tell the agent to spawn a pipeline for this room.
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                f"{settings.agent_dispatch_url}/dispatch",
                json={"room": room},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Agent dispatch failed: {err}", err=str(exc))
            raise HTTPException(status_code=503, detail="Agent unavailable") from exc

    return SessionResponse(
        url=settings.livekit_public_url,
        token=token,
        room=room,
        identity=identity,
    )

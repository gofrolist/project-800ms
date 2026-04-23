"""project-800ms API.

Endpoints:
    GET  /health            — liveness
    POST /v1/sessions       — mint a LiveKit caller token (X-API-Key auth)
    GET  /v1/sessions/{room} — fetch session details

Docs surface (all publicly reachable):
    GET  /docs              — Swagger UI
    GET  /redoc             — ReDoc
    GET  /reference         — Scalar API Reference
    GET  /openapi.json      — raw OpenAPI 3.1 spec
"""

from __future__ import annotations

import logging
import os

import errors
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from observability import configure_json_logging, install_metrics
from rate_limit import limiter
from request_id import RequestIdMiddleware
from routes.admin import router as admin_router
from routes.engines import router as engines_router
from routes.sessions import router as sessions_router
from routes.transcripts import internal_router as transcripts_internal_router
from routes.transcripts import v1_router as transcripts_v1_router
from routes.usage import router as usage_router
from routes.webhooks import router as webhooks_router
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from settings import settings

# Install JSON logging before anything else logs, so early startup
# records (Settings() resolution, module imports) still come through
# structured. Level comes from LOG_LEVEL env, default INFO.
configure_json_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

logger = logging.getLogger("project-800ms.api")


API_DESCRIPTION = """
Real-time voice-agent API for game clients.

## Authentication

Every `/v1/*` endpoint requires an **`X-API-Key`** header. Request keys
from your operations contact. Keys are tenant-scoped; revoking a key
does not affect other tenants.

## Error envelope

All `/v1/*` error responses share a common shape:

```json
{
  "error": {
    "code": "unauthenticated",
    "message": "Missing or malformed X-API-Key header",
    "request_id": "01HQXZ3Y4A5TGJ6GGC8CVKWE0N"
  }
}
```

`request_id` is a ULID returned in the `X-Request-ID` header of every
response. Include it in support tickets so we can correlate logs.

## Rate limits

`/v1/*` endpoints are rate-limited per tenant at 60 requests/minute by
default. Exceeding the budget returns **429 rate_limited**.
"""


app = FastAPI(
    title="project-800ms API",
    version="0.2.0",
    summary="Voice-agent session API.",
    description=API_DESCRIPTION,
    contact={
        "name": "project-800ms",
        "url": "https://github.com/gofrolist/project-800ms",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=[
        {
            "name": "sessions",
            "description": "Open and inspect voice sessions.",
        },
        {
            "name": "transcripts",
            "description": "Read persisted utterances for past sessions.",
        },
        {
            "name": "usage",
            "description": "Per-tenant daily audio consumption.",
        },
        {
            "name": "system",
            "description": "Liveness and service metadata.",
        },
        {
            "name": "admin",
            "description": (
                "Operator-only tenant + API-key management. Requires "
                "`X-Admin-Key` (not `X-API-Key`)."
            ),
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
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
        # CSP is permissive for /docs, /redoc, /reference — they load
        # Swagger / ReDoc / Scalar assets from jsdelivr. Every other route
        # returns JSON so the extra sources are never used.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self' wss:; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://cdn.jsdelivr.net; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "worker-src 'self' blob:",
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
    # Methods intentionally limited to GET + POST + DELETE. The /v1/admin/*
    # surface uses PATCH but is server-only (operator tooling, curl/terraform)
    # and is not designed to be reachable from browser clients — omitting
    # PATCH here is defense-in-depth that blocks any future browser-driven
    # admin UI from bypassing the server-only boundary. DELETE is allowed so
    # the demo web UI's "hang up" button can force-close a session (see
    # DELETE /v1/sessions/{room}). Extend deliberately, not reflexively, if
    # that constraint ever needs to change.
    allow_methods=["GET", "POST", "DELETE"],
    # Client sends Content-Type + X-API-Key + optional X-Request-ID.
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


# ─── Routes ───────────────────────────────────────────────────────────────
app.include_router(sessions_router)
app.include_router(engines_router)
app.include_router(webhooks_router)
app.include_router(transcripts_v1_router)
app.include_router(transcripts_internal_router)
app.include_router(usage_router)
app.include_router(admin_router)


# ─── Observability ───────────────────────────────────────────────────────
# /metrics + the Prometheus middleware land last so they observe the
# final response (after errors.install's exception handlers have
# converted APIErrors to JSONResponses).
install_metrics(app)


@app.get("/health", tags=["system"], summary="Liveness probe")
@limiter.limit("60/minute")
def health(request: Request) -> dict[str, str]:
    """Return `{"status": "ok"}` whenever the process is up.

    Does not exercise the database or the downstream agent — use this for
    TCP-level health checks and the `/v1/sessions` create path for deep
    end-to-end validation.
    """
    return {"status": "ok"}


# ─── Scalar docs UI ───────────────────────────────────────────────────────
# Scalar is a Swagger/ReDoc alternative with a better default UX. We serve
# its HTML from a CDN — no package dep needed. The spec URL points at our
# existing /openapi.json.
_SCALAR_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>project-800ms API Reference</title>
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{"theme":"default","layout":"modern"}'
    ></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""


@app.get(
    "/reference",
    include_in_schema=False,
    response_class=HTMLResponse,
    summary="Scalar API Reference",
)
def scalar_reference() -> HTMLResponse:
    return HTMLResponse(_SCALAR_HTML)

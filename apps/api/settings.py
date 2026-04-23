"""Runtime settings. Read once at startup, immutable thereafter."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # LiveKit public URL — what the browser connects to. Defaults to the
    # dev single-box setup; set to wss://livekit.yourdomain.com in prod.
    livekit_public_url: str = "ws://localhost:7880"
    # LiveKit server-side URL the API uses for RoomService calls (today:
    # delete_room on DELETE /v1/sessions/{room}). In a docker-compose
    # single-box deploy this is the internal service hostname
    # (ws://livekit:7880) because the API container can't reach the
    # public `localhost:7880` mapping from inside the container network.
    # Falls back to ``livekit_public_url`` when empty, which is correct
    # for prod where both the browser and the API hit the same routable
    # hostname.
    livekit_url: str = ""
    # API key is a short identifier; the secret signs JWTs via HMAC-SHA256.
    # LiveKit recommends a 32+ char secret for HMAC strength. Empty values
    # would silently produce syntactically valid but functionally invalid JWTs.
    livekit_api_key: Annotated[str, Field(min_length=3)]
    livekit_api_secret: Annotated[str, Field(min_length=32)]

    # Caller config. Token TTL is intentionally short — the browser only
    # needs the token long enough to complete the WebRTC handshake. Once
    # the participant is in the room, LiveKit's session lifecycle takes
    # over. A leaked token can be replayed for the TTL window, so keep it
    # tight.
    session_ttl_seconds: int = 15 * 60  # 15m

    # Internal agent dispatch endpoint. The API calls this to tell the agent
    # to spawn a pipeline for a new room.
    agent_dispatch_url: str = "http://agent:8001"

    # CORS. Set CORS_ALLOWED_ORIGINS to a comma-separated list of explicit
    # origins, e.g. "https://app.example.com,https://example.com". Wildcard
    # is accepted only in local dev (tests + docker-compose.yml default);
    # production deployments MUST override. Per-tenant origin pinning in
    # tenants.allowed_origins is the real access-control boundary — this
    # list is a belt-and-braces second layer.
    cors_allowed_origins: list[str] = ["*"]

    # Database — async SQLAlchemy URL (postgresql+asyncpg://...). The init
    # container runs Alembic migrations before the API starts, so by the
    # time this engine connects the schema is guaranteed current.
    database_url: str = "postgresql+asyncpg://voice:voice@postgres:5432/voice"

    # Shared secret for server-to-server calls from the agent to /internal/*
    # endpoints (today: transcript writes). The agent presents this value
    # as `X-Internal-Token`. Empty disables the endpoint — useful for
    # tenants that don't need transcript persistence.
    agent_internal_token: str = ""

    # Master key for /v1/admin/* endpoints (create / rotate tenants + keys).
    # Empty disables the admin surface entirely — preferred in deployments
    # that provision out-of-band (e.g. via direct DB access). Raw key is
    # only sent over the wire; no session cookie / refresh flow, so
    # exposure should be treated as total compromise — rotate by changing
    # this env var and redeploying.
    admin_api_key: str = ""

    # How long a successful X-API-Key lookup stays in process-local cache.
    # Shorter = faster revocation propagation, more DB load. 60s is a
    # reasonable tradeoff for a single-replica API.
    tenant_cache_ttl_seconds: int = 60

    # CIDR ranges whose TCP peer we trust to pass a real-client IP via
    # X-Forwarded-For. Default is RFC1918 172.16.0.0/12 — the Docker
    # bridge range in our compose deploy, where Caddy is the only thing
    # calling the API container. If the deploy topology changes (non-
    # Docker proxy, ALB, etc.) override this with the proxy's actual
    # subnet. An empty list disables XFF trust entirely — _real_ip will
    # always fall back to the TCP peer address, which is safe when the
    # API is reached directly without a proxy.
    trusted_proxy_cidrs: list[str] = ["172.16.0.0/12"]

    # IP-based rate limit for /v1/admin/* (defense-in-depth against online
    # brute-force of admin_api_key). 60/minute is generous for a human
    # operator and still infeasible against a 256-bit key. Raise for ops
    # workflows that burst from one egress (Terraform for_each, CI seeding,
    # bulk key rotation) — e.g. 300 or 600. Values are advisory; the real
    # boundary remains secrets.compare_digest on the key itself.
    admin_ip_rate_per_minute: int = 60

    # IP-based rate limit for /v1/livekit-webhook. Guards against JWT-flood
    # CPU burn on invalid signatures. 1000/minute is well above normal
    # LiveKit event rates from a single egress IP; raise if your deploy
    # runs many concurrent rooms behind one LiveKit instance and starts
    # seeing legitimate 429s on the webhook path.
    webhook_ip_rate_per_minute: int = 1000


settings = Settings()  # type: ignore[call-arg]

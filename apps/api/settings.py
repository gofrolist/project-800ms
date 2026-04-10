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

    # CORS. Default is permissive for local dev (browser at :5173 hits API
    # at :8000). In prod, set CORS_ALLOWED_ORIGINS to a comma-separated list
    # of explicit origins, e.g. "https://app.example.com,https://example.com".
    cors_allowed_origins: list[str] = ["*"]


settings = Settings()  # type: ignore[call-arg]

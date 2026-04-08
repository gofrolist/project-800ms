"""Runtime settings. Read once at startup, immutable thereafter."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # LiveKit. Server talks to LiveKit over the internal docker network;
    # browsers need a public URL. Both default to the dev single-box setup.
    livekit_url: str = "ws://livekit:7880"
    livekit_public_url: str = "ws://localhost:7880"
    livekit_api_key: str
    livekit_api_secret: str

    # Caller config
    demo_room: str = "demo"
    session_ttl_seconds: int = 60 * 60  # 1h


settings = Settings()  # type: ignore[call-arg]

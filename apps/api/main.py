"""project-800ms API.

Endpoints:
    GET  /health      — liveness
    POST /sessions    — mint a LiveKit caller token for the demo room
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api as lkapi
from pydantic import BaseModel

from settings import settings

app = FastAPI(title="project-800ms API")

# Dev-wide CORS — tighten for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionResponse(BaseModel):
    """What the browser needs to join a call."""

    url: str       # LiveKit ws URL (public)
    token: str     # JWT for this caller
    room: str      # Room name the agent is sitting in
    identity: str  # Caller identity embedded in the token


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Mint a caller token for the demo room.

    The agent is already sitting in `settings.demo_room`, so the caller
    just joins and starts talking. Auth/user binding lands in a follow-up.
    """
    identity = f"user-{uuid.uuid4().hex[:8]}"
    token = (
        lkapi.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(
            lkapi.VideoGrants(
                room_join=True,
                room=settings.demo_room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(settings.session_ttl_seconds)
        .to_jwt()
    )
    return SessionResponse(
        url=settings.livekit_public_url,
        token=token,
        room=settings.demo_room,
        identity=identity,
    )

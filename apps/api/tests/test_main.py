"""HTTP-level tests for the api endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app, limiter
from settings import settings


@pytest.fixture(autouse=True)
def _reset_limiter() -> Iterator[None]:
    """Reset the in-memory rate-limit storage before each test.

    slowapi's MemoryStorage is process-global; without resetting, the
    counters from one test bleed into the next and a test that does 5
    POSTs will trip the next test's first request.
    """
    limiter.reset()
    yield
    limiter.reset()


def _mock_dispatch() -> AsyncMock:
    """Return a mock that simulates a successful agent dispatch response."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.fixture(autouse=True)
def _mock_agent_dispatch():
    """Mock httpx.AsyncClient so tests don't need a running agent."""
    with patch("main.httpx.AsyncClient", return_value=_mock_dispatch()):
        yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


class TestSecurityHeaders:
    def test_default_headers_present(self, client: TestClient) -> None:
        res = client.get("/health")
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        assert res.headers["Referrer-Policy"] == "no-referrer"
        assert "max-age=" in res.headers["Strict-Transport-Security"]


class TestCreateSession:
    def test_returns_session_envelope(self, client: TestClient) -> None:
        res = client.post("/sessions")
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"url", "token", "room", "identity"}
        assert body["url"] == settings.livekit_public_url
        assert body["room"].startswith("room-")
        assert body["identity"].startswith("user-")

    def test_token_is_signed_with_configured_secret(self, client: TestClient) -> None:
        res = client.post("/sessions")
        body = res.json()
        # Decode without verification first to read the identity claim, then
        # verify against the configured secret to prove signing worked.
        unverified = jwt.decode(body["token"], options={"verify_signature": False})
        assert unverified["sub"] == body["identity"]

        decoded = jwt.decode(
            body["token"],
            settings.livekit_api_secret,
            algorithms=["HS256"],
        )
        assert decoded["video"]["room"] == body["room"]
        assert decoded["video"]["roomJoin"] is True

    def test_each_call_mints_unique_identity(self, client: TestClient) -> None:
        identities = {client.post("/sessions").json()["identity"] for _ in range(5)}
        assert len(identities) == 5

    def test_each_call_creates_unique_room(self, client: TestClient) -> None:
        rooms = {client.post("/sessions").json()["room"] for _ in range(5)}
        assert len(rooms) == 5


class TestRateLimiting:
    def test_sessions_limited_to_5_per_minute(self, client: TestClient) -> None:
        # First 5 requests succeed, the 6th must be rejected with 429.
        for _ in range(5):
            assert client.post("/sessions").status_code == 200
        res = client.post("/sessions")
        assert res.status_code == 429

    def test_health_limited_to_60_per_minute(self, client: TestClient) -> None:
        # 60 requests fit within the window; the 61st must be 429.
        for _ in range(60):
            assert client.get("/health").status_code == 200
        res = client.get("/health")
        assert res.status_code == 429

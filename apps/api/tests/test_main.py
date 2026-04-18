"""App-level smoke tests — health, security headers, rate limiting.

Endpoint-specific tests live alongside the routes (test_v1_sessions.py,
test_auth.py, test_errors.py). This file covers the app-wide middleware
surface that doesn't fit in a single route's test file.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from main import app
from rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_limiter() -> Iterator[None]:
    """Reset slowapi's in-memory buckets between tests.

    Without this, the 60/minute bucket leaks across tests and the first
    /health call in test N starts at a random offset.
    """
    limiter.reset()
    yield
    limiter.reset()


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
        assert "default-src 'self'" in res.headers["Content-Security-Policy"]


class TestRateLimiting:
    def test_health_limited_to_60_per_minute(self, client: TestClient) -> None:
        for _ in range(60):
            assert client.get("/health").status_code == 200
        res = client.get("/health")
        assert res.status_code == 429

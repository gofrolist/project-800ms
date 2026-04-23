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


class TestLivekitUrlStartupWarning:
    """Pins the startup-time warning that fires when a non-development
    deploy leaves LIVEKIT_URL unset. Without this signal, misconfigured
    prod/staging deploys only manifest as zombie LiveKit rooms surfaced
    via support tickets — the DELETE /v1/sessions/{room} handler swallows
    the downstream failure so nothing in HTTP metrics points at the root
    cause.

    The warn check is extracted into a named function specifically so
    tests can exercise it directly instead of reloading the main module.

    Log capture uses a direct handler attached to the project logger
    rather than pytest's ``caplog`` fixture: ``configure_json_logging``
    (called at main.py import time) clears the root logger's handlers,
    which races with caplog's setup under some test orderings and leaves
    the fixture empty. An explicit handler on the specific logger
    sidesteps that interaction entirely.
    """

    @staticmethod
    def _capture_warnings(fn) -> list[str]:
        """Invoke ``fn()`` with a handler attached to the api logger and
        return the list of log messages it emitted. Cleans up the
        handler on exit so subsequent tests aren't polluted.

        Note: pytest's internal log-capture machinery sets
        ``logger.disabled = True`` on cached loggers between tests,
        which causes ``logger.warning`` to silently no-op. We flip it
        back to False around the call and restore afterwards — attaching
        an additional handler alone isn't enough when the logger itself
        is disabled.
        """
        import logging

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        lg = logging.getLogger("project-800ms.api")
        lg.addHandler(handler)
        previous_level = lg.level
        previous_disabled = lg.disabled
        lg.setLevel(logging.WARNING)
        lg.disabled = False
        try:
            fn()
        finally:
            lg.removeHandler(handler)
            lg.setLevel(previous_level)
            lg.disabled = previous_disabled
        return [r.getMessage() for r in records]

    def test_warns_when_url_missing_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from main import _warn_if_livekit_url_missing_in_non_local_env  # noqa: PLC0415
        from settings import settings  # noqa: PLC0415

        monkeypatch.setattr(settings, "livekit_url", "")
        monkeypatch.setenv("ENV", "production")

        messages = self._capture_warnings(_warn_if_livekit_url_missing_in_non_local_env)
        assert any("LIVEKIT_URL is not set" in m for m in messages), (
            f"expected LIVEKIT_URL warning, got: {messages}"
        )

    def test_no_warning_in_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from main import _warn_if_livekit_url_missing_in_non_local_env  # noqa: PLC0415
        from settings import settings  # noqa: PLC0415

        monkeypatch.setattr(settings, "livekit_url", "")
        monkeypatch.setenv("ENV", "development")

        messages = self._capture_warnings(_warn_if_livekit_url_missing_in_non_local_env)
        assert not any("LIVEKIT_URL" in m for m in messages)

    def test_no_warning_when_env_unset_defaults_to_development(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENV variable not set at all — os.environ.get defaults to
        'development', so no warning fires. Pins that local dev
        (missing env) is the silent-OK path, not a noisy one."""
        from main import _warn_if_livekit_url_missing_in_non_local_env  # noqa: PLC0415
        from settings import settings  # noqa: PLC0415

        monkeypatch.setattr(settings, "livekit_url", "")
        monkeypatch.delenv("ENV", raising=False)

        messages = self._capture_warnings(_warn_if_livekit_url_missing_in_non_local_env)
        assert not any("LIVEKIT_URL" in m for m in messages)

    def test_no_warning_when_url_set_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from main import _warn_if_livekit_url_missing_in_non_local_env  # noqa: PLC0415
        from settings import settings  # noqa: PLC0415

        monkeypatch.setattr(settings, "livekit_url", "ws://livekit:7880")
        monkeypatch.setenv("ENV", "production")

        messages = self._capture_warnings(_warn_if_livekit_url_missing_in_non_local_env)
        assert not any("LIVEKIT_URL" in m for m in messages)

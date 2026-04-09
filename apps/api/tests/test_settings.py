"""Validation tests for the Settings model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from settings import Settings

# 32-char filler secret used across tests. Real values come from env in prod.
VALID_SECRET = "x" * 32


class TestSettingsValidation:
    def test_valid_settings_loads(self) -> None:
        s = Settings(
            livekit_api_key="abc",
            livekit_api_secret=VALID_SECRET,
        )
        assert s.livekit_api_key == "abc"
        assert s.livekit_api_secret == VALID_SECRET
        assert s.demo_room == "demo"
        assert s.session_ttl_seconds == 15 * 60
        assert s.cors_allowed_origins == ["*"]

    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            Settings(livekit_api_key="", livekit_api_secret=VALID_SECRET)
        assert "livekit_api_key" in str(exc.value)

    def test_short_api_key_rejected(self) -> None:
        # min_length=3 — shorter is rejected.
        with pytest.raises(ValidationError) as exc:
            Settings(livekit_api_key="ab", livekit_api_secret=VALID_SECRET)
        assert "livekit_api_key" in str(exc.value)

    def test_empty_api_secret_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            Settings(livekit_api_key="key", livekit_api_secret="")
        assert "livekit_api_secret" in str(exc.value)

    def test_short_api_secret_rejected(self) -> None:
        # min_length=32 — anything shorter (e.g. 31 chars) is rejected.
        with pytest.raises(ValidationError) as exc:
            Settings(livekit_api_key="key", livekit_api_secret="x" * 31)
        assert "livekit_api_secret" in str(exc.value)

    def test_cors_origins_overridable(self) -> None:
        s = Settings(
            livekit_api_key="key",
            livekit_api_secret=VALID_SECRET,
            cors_allowed_origins=["https://example.com", "https://app.example.com"],
        )
        assert s.cors_allowed_origins == [
            "https://example.com",
            "https://app.example.com",
        ]

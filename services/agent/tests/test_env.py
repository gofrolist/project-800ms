"""Unit tests for the env helper."""

from __future__ import annotations

import pytest

from env import MissingEnvError, require_env


class TestRequireEnv:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_VAR", "hello")
        assert require_env("TEST_VAR") == "hello"

    def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_VAR", raising=False)
        assert require_env("TEST_VAR", "fallback") == "fallback"

    def test_set_value_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_VAR", "real")
        assert require_env("TEST_VAR", "fallback") == "real"

    def test_missing_with_no_default_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TEST_VAR", raising=False)
        with pytest.raises(MissingEnvError, match="TEST_VAR"):
            require_env("TEST_VAR")

    def test_empty_string_treated_as_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An empty credential is almost always a misconfiguration; reject it
        # the same way we reject an unset variable.
        monkeypatch.setenv("TEST_VAR", "")
        with pytest.raises(MissingEnvError, match="TEST_VAR"):
            require_env("TEST_VAR")

    def test_empty_default_also_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TEST_VAR", raising=False)
        with pytest.raises(MissingEnvError, match="TEST_VAR"):
            require_env("TEST_VAR", "")

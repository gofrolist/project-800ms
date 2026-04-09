"""Tests for the Russian-only language detector and stateful router."""

from __future__ import annotations

import pytest

from lang import Language, LanguageRouter, detect_language


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text",
        [
            "Привет",
            "Привет, как дела?",
            "Спасибо, всё хорошо.",
            "Я",
            "Расскажи мне сказку",
        ],
    )
    def test_pure_russian(self, text: str) -> None:
        assert detect_language(text) == Language.RU

    @pytest.mark.parametrize(
        "text",
        [
            "Hello",
            "OK",
            "The quick brown fox",
        ],
    )
    def test_latin_text_still_russian(self, text: str) -> None:
        # Latin text is treated as Russian since that's our only language.
        assert detect_language(text) == Language.RU

    @pytest.mark.parametrize(
        "text",
        ["", " ", ".", "...", "!?", "  \n  ", "123", "$5.00"],
    )
    def test_no_signal_returns_russian(self, text: str) -> None:
        assert detect_language(text) == Language.RU

    @pytest.mark.parametrize(
        "text",
        [
            "因为鱼喜欢水",
            "你好世界",
            "Hello 你好",
        ],
    )
    def test_cjk_returns_none(self, text: str) -> None:
        assert detect_language(text) is None

    def test_language_string_value(self) -> None:
        assert Language.RU == "ru"


class TestLanguageRouter:
    def test_starts_in_russian(self) -> None:
        r = LanguageRouter()
        assert r.last == Language.RU

    def test_pure_russian_stream(self) -> None:
        r = LanguageRouter()
        for chunk in ["Привет", ", ", "как", " дела", "?"]:
            assert r.route(chunk) == Language.RU
        assert r.last == Language.RU

    def test_punctuation_inherits_russian(self) -> None:
        r = LanguageRouter()
        assert r.route("Привет") == Language.RU
        assert r.route(".") == Language.RU

    def test_cjk_does_not_update_last(self) -> None:
        r = LanguageRouter()
        assert r.route("Привет") == Language.RU
        assert r.route("你好") is None
        assert r.last == Language.RU

    def test_independent_routers_dont_share_state(self) -> None:
        a = LanguageRouter()
        b = LanguageRouter()
        a.route("Привет")
        assert a.last == Language.RU
        assert b.last == Language.RU

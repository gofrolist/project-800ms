"""Tests for the bilingual language detector and stateful router."""

from __future__ import annotations

import pytest

from lang import Language, LanguageRouter, detect_language


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text",
        [
            "Hello",
            "Hello, how are you?",
            "I'm doing well, thanks.",
            "I",  # single letter
            "OK",
            "The quick brown fox",
        ],
    )
    def test_pure_english(self, text: str) -> None:
        assert detect_language(text) == Language.EN

    @pytest.mark.parametrize(
        "text",
        [
            "Привет",
            "Привет, как дела?",
            "Спасибо, всё хорошо.",
            "Я",  # single letter
            "Расскажи мне сказку",
        ],
    )
    def test_pure_russian(self, text: str) -> None:
        assert detect_language(text) == Language.RU

    def test_mixed_routes_to_russian(self) -> None:
        # Any cyrillic char → RU. Russian Piper handles latin filler
        # better than English Piper handles cyrillic.
        assert detect_language("Hello, мир") == Language.RU
        assert detect_language("привет world") == Language.RU
        assert detect_language("foo бар baz") == Language.RU

    @pytest.mark.parametrize(
        "text",
        ["", " ", ".", "...", "!?", "  \n  ", "123", "$5.00"],
    )
    def test_no_signal_uses_fallback(self, text: str) -> None:
        # No alphabetic content → use the caller-supplied fallback.
        assert detect_language(text, fallback=Language.EN) == Language.EN
        assert detect_language(text, fallback=Language.RU) == Language.RU

    def test_default_fallback_is_english(self) -> None:
        assert detect_language(".") == Language.EN
        assert detect_language("") == Language.EN

    def test_digits_alone_use_fallback(self) -> None:
        assert detect_language("42", fallback=Language.RU) == Language.RU
        assert detect_language("3.14159", fallback=Language.EN) == Language.EN

    def test_language_string_value(self) -> None:
        # str-mixin: enum members are usable wherever a plain string is
        # expected (env vars, frame metadata, log lines).
        assert Language.EN == "en"
        assert Language.RU == "ru"


class TestLanguageRouter:
    def test_starts_in_english(self) -> None:
        r = LanguageRouter()
        assert r.last == Language.EN

    def test_default_overrideable(self) -> None:
        r = LanguageRouter(default=Language.RU)
        assert r.last == Language.RU

    def test_pure_english_stream(self) -> None:
        r = LanguageRouter()
        for chunk in ["Hello", ", ", "how", " are", " you", "?"]:
            assert r.route(chunk) == Language.EN
        assert r.last == Language.EN

    def test_pure_russian_stream(self) -> None:
        r = LanguageRouter()
        for chunk in ["Привет", ", ", "как", " дела", "?"]:
            assert r.route(chunk) == Language.RU
        assert r.last == Language.RU

    def test_punctuation_inherits_previous(self) -> None:
        # Realistic LLM streaming pattern: words then a separate "."
        # frame. The "." has no language signal of its own — it should
        # follow whichever language the surrounding tokens were.
        r = LanguageRouter()
        assert r.route("Привет") == Language.RU
        assert r.route(".") == Language.RU  # inherits RU
        assert r.route("Hello") == Language.EN
        assert r.route(".") == Language.EN  # now inherits EN

    def test_language_switch_mid_stream(self) -> None:
        # The agent answers an EN question, then a RU one. State
        # transitions cleanly without leaking the previous language.
        r = LanguageRouter()
        assert r.route("I'm doing well.") == Language.EN
        assert r.route("Спасибо!") == Language.RU
        assert r.route("...") == Language.RU
        assert r.route("Sure!") == Language.EN

    def test_independent_routers_dont_share_state(self) -> None:
        a = LanguageRouter()
        b = LanguageRouter()
        a.route("Привет")
        assert a.last == Language.RU
        assert b.last == Language.EN  # unchanged

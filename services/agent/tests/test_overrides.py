"""Tests for per-session override parsing + system-prompt composition.

Import-safe — overrides.py has no pipecat / CUDA deps.
"""

from __future__ import annotations

from overrides import (
    DEFAULT_GREETINGS_BY_LANGUAGE,
    VOICE_RULES_BY_LANGUAGE,
    PerSessionOverrides,
    build_system_prompt,
    resolve_greeting,
)


class TestFromDispatch:
    def test_empty_body_yields_all_none(self) -> None:
        o = PerSessionOverrides.from_dispatch({})
        assert o == PerSessionOverrides()
        assert o.effective_language == "ru"

    def test_reads_known_keys(self) -> None:
        o = PerSessionOverrides.from_dispatch(
            {
                "persona": {"name": "Grisha"},
                "voice": "ru_RU-denis-medium",
                "language": "en",
                "llm_model": "llama-3.3-70b-versatile",
                "user_id": "player-42",
                "npc_id": "merchant",
                "context": {"inventory": ["sword"]},
            }
        )
        assert o.persona == {"name": "Grisha"}
        assert o.voice == "ru_RU-denis-medium"
        assert o.language == "en"
        assert o.llm_model == "llama-3.3-70b-versatile"
        assert o.user_id == "player-42"
        assert o.npc_id == "merchant"
        assert o.context == {"inventory": ["sword"]}
        assert o.effective_language == "en"

    def test_ignores_unknown_keys(self) -> None:
        o = PerSessionOverrides.from_dispatch({"bogus": 42, "voice": "v"})
        assert o.voice == "v"

    def test_empty_strings_dropped(self) -> None:
        """Empty strings are indistinguishable from missing — drop them so
        downstream callers can rely on `None | non-empty str`."""
        o = PerSessionOverrides.from_dispatch({"voice": "", "language": "", "persona": {}})
        assert o.voice is None
        assert o.language is None
        assert o.persona is None

    def test_wrong_types_dropped(self) -> None:
        o = PerSessionOverrides.from_dispatch(
            {"voice": 123, "persona": "not a dict", "context": ["also not"]}
        )
        assert o.voice is None
        assert o.persona is None
        assert o.context is None


class TestBuildSystemPrompt:
    def test_no_persona_returns_rules(self) -> None:
        assert build_system_prompt(None) == VOICE_RULES_BY_LANGUAGE["ru"]

    def test_language_en_switches_rules(self) -> None:
        assert build_system_prompt(None, "en") == VOICE_RULES_BY_LANGUAGE["en"]

    def test_unknown_language_falls_back_to_default(self) -> None:
        assert build_system_prompt(None, "fr") == VOICE_RULES_BY_LANGUAGE["ru"]

    def test_explicit_system_prompt_wins(self) -> None:
        out = build_system_prompt({"system_prompt": "Be Yoda.", "name": "ignored"})
        assert out == "Be Yoda."

    def test_system_prompt_empty_string_ignored(self) -> None:
        """Empty-string system_prompt shouldn't silently replace the whole
        prompt with nothing."""
        out = build_system_prompt({"system_prompt": "   ", "name": "X"})
        assert "X" in out
        assert VOICE_RULES_BY_LANGUAGE["ru"] in out

    def test_composes_name_backstory_style(self) -> None:
        out = build_system_prompt(
            {
                "name": "Гриша",
                "backstory": "Угрюмый торговец из Владивостока.",
                "style": "Ворчливый, короткие ответы.",
            }
        )
        assert "Гриша" in out
        assert "Владивостока" in out
        assert "Ворчливый" in out
        # Voice rules are always appended.
        assert VOICE_RULES_BY_LANGUAGE["ru"] in out

    def test_only_name_still_works(self) -> None:
        out = build_system_prompt({"name": "Solo"})
        assert out.startswith("Solo")
        assert VOICE_RULES_BY_LANGUAGE["ru"] in out

    def test_non_string_persona_fields_ignored(self) -> None:
        out = build_system_prompt({"name": 42, "backstory": None, "style": ["x"]})
        assert out == VOICE_RULES_BY_LANGUAGE["ru"]


class TestResolveGreeting:
    def test_default_russian(self) -> None:
        assert resolve_greeting(None) == DEFAULT_GREETINGS_BY_LANGUAGE["ru"]

    def test_default_english(self) -> None:
        assert resolve_greeting(None, "en") == DEFAULT_GREETINGS_BY_LANGUAGE["en"]

    def test_custom_greeting_wins(self) -> None:
        assert resolve_greeting({"greeting": "Здорово!"}) == "Здорово!"

    def test_empty_custom_greeting_falls_back(self) -> None:
        assert resolve_greeting({"greeting": "   "}) == DEFAULT_GREETINGS_BY_LANGUAGE["ru"]

    def test_non_string_greeting_ignored(self) -> None:
        assert resolve_greeting({"greeting": 42}, "en") == DEFAULT_GREETINGS_BY_LANGUAGE["en"]

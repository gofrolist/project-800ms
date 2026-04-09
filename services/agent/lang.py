"""Tiny language detector for routing TTS to the right Piper voice.

We only support English and Russian today, so this is intentionally a
hand-rolled cyrillic-vs-latin classifier rather than a heavyweight
language-detection library:

- Pure stdlib, ~5 microseconds per call → no impact on the latency budget.
- Zero dependencies → no extra wheels to ship in the Docker image.
- Deterministic on short LLM-streamed chunks (where probabilistic
  detectors trained on full sentences tend to flip-flop).

When we add a third language, swap this for `lingua` (or similar) and
keep the same `detect_language(text, fallback)` signature.
"""

from __future__ import annotations

from enum import Enum

CYRILLIC_START = "\u0400"
CYRILLIC_END = "\u04ff"


class Language(str, Enum):
    """Supported TTS output languages.

    `str`-mixin so the value compares equal to "en" / "ru" — useful when
    serialising into env vars, logs, or pipecat frame metadata.
    """

    EN = "en"
    RU = "ru"


def detect_language(text: str, fallback: Language = Language.EN) -> Language:
    """Classify a text chunk as English or Russian.

    Heuristic, in order of precedence:
        1. Any Cyrillic character → Russian. (Mixed strings like
           "Hello мир" route to the Russian voice; Piper RU pronounces
           Latin filler reasonably and we'd rather err that way than
           hand "мир" to a Russian-voice with English phonemes.)
        2. Any Latin alphabet character → English.
        3. Otherwise (punctuation, digits, whitespace only) → fallback.
           This covers LLM-streamed chunks like "." or "  " that carry
           no language signal of their own — the caller should pass the
           previously-detected language as `fallback` so the chunk
           routes to the same voice as its neighbours.
    """
    has_cyrillic = False
    has_latin = False
    for ch in text:
        if CYRILLIC_START <= ch <= CYRILLIC_END:
            has_cyrillic = True
            break  # cyrillic wins immediately
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            has_latin = True
    if has_cyrillic:
        return Language.RU
    if has_latin:
        return Language.EN
    return fallback


class LanguageRouter:
    """Stateful wrapper around `detect_language`.

    Remembers the last language it returned so that subsequent
    language-signal-free chunks (punctuation, digits, whitespace)
    inherit it instead of always defaulting to English. Pull this out
    of the FrameProcessor so the routing logic can be unit-tested
    without pipecat's frame machinery.
    """

    def __init__(self, default: Language = Language.EN) -> None:
        self._last = default

    @property
    def last(self) -> Language:
        return self._last

    def route(self, text: str) -> Language:
        """Detect the language of `text`, remember it, return it."""
        self._last = detect_language(text, fallback=self._last)
        return self._last

"""Tiny language detector — Russian-only for now.

Hand-rolled cyrillic classifier:
- Pure stdlib, ~5 microseconds per call → no impact on the latency budget.
- Zero dependencies → no extra wheels to ship in the Docker image.
- Deterministic on short LLM-streamed chunks.
"""

from __future__ import annotations

from enum import Enum

CYRILLIC_START = "\u0400"
CYRILLIC_END = "\u04ff"

# CJK Unified Ideographs — some models occasionally leak Chinese.
CJK_RANGES = (
    ("\u4e00", "\u9fff"),   # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),   # CJK Extension A
    ("\u3000", "\u303f"),   # CJK Symbols and Punctuation
)


class Language(str, Enum):
    """Supported TTS output languages."""

    RU = "ru"


def _is_cjk(ch: str) -> bool:
    return any(start <= ch <= end for start, end in CJK_RANGES)


def detect_language(text: str, fallback: Language = Language.RU) -> Language | None:
    """Classify a text chunk as Russian or unsupported.

    Returns ``None`` when the text contains CJK characters that
    the Piper voice can't pronounce.

    Heuristic:
        1. Any CJK character → ``None`` (unsupported script).
        2. Otherwise → Russian (our only language).
    """
    for ch in text:
        if _is_cjk(ch):
            return None
    return Language.RU


class LanguageRouter:
    """Stateful wrapper around `detect_language`.

    Keeps the same interface for compatibility with the pipeline,
    but always returns Russian for supported text.
    """

    def __init__(self, default: Language = Language.RU) -> None:
        self._last = default

    @property
    def last(self) -> Language:
        return self._last

    def route(self, text: str) -> Language | None:
        """Detect the language of `text`, remember it, return it.

        Returns ``None`` for unsupported scripts (CJK).
        """
        detected = detect_language(text, fallback=self._last)
        if detected is not None:
            self._last = detected
        return detected

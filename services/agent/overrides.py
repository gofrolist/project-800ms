"""Per-session overrides from the API's /dispatch payload.

Lives in its own module so it can be unit-tested without importing pipecat
(which pulls in CUDA). `pipeline.py` takes a `PerSessionOverrides` and
reshapes the STT / LLM / TTS services before constructing the task.

Persona shape:
    {
      "name":          str,  # character name, used in intro + greeting
      "backstory":     str,  # long-form background
      "style":         str,  # speaking style hint
      "system_prompt": str,  # full override — skips all composition
      "greeting":      str,  # first utterance when the caller joins
    }

Any subset is valid. Unknown keys are ignored — we don't want a brittle
schema for game clients still iterating on persona content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Hard style constraints the TTS output depends on. Kept separate from the
# persona-specific content so callers can swap the character while keeping
# the voice-safe formatting rules intact.
VOICE_RULES_BY_LANGUAGE: dict[str, str] = {
    "ru": (
        "Твои ответы озвучиваются через синтез речи, поэтому строго соблюдай "
        "правила: отвечай коротко, 1-3 предложения, разговорный тон. "
        "Никогда не используй маркдаун, звёздочки, списки, тире, скобки, "
        "кавычки, эмодзи и специальные символы. Используй только обычные "
        "слова и базовую пунктуацию. Всегда отвечай только на русском языке. "
        "Все имена и названия транслитерируй на русский."
    ),
    "en": (
        "Your answers are spoken through a voice synthesizer, so follow "
        "these rules strictly: answer in 1-3 short conversational sentences. "
        "Never use markdown, asterisks, lists, dashes, parentheses, quotes, "
        "emoji, or special symbols. Use only regular words and basic "
        "punctuation. Always answer only in English."
    ),
}

DEFAULT_GREETINGS_BY_LANGUAGE: dict[str, str] = {
    "ru": "Привет! Чем могу помочь?",
    "en": "Hi! How can I help?",
}

DEFAULT_LANGUAGE = "ru"

# Mirrors pipeline._VALID_TTS_ENGINES; duplicated here so parsing the
# dispatch body doesn't require importing pipeline (which pulls in
# pipecat + CUDA on the import path).
_VALID_TTS_ENGINES: frozenset[str] = frozenset({"piper", "silero", "qwen3", "xtts"})


@dataclass(frozen=True)
class PerSessionOverrides:
    """Optional knobs supplied by the API for a single room."""

    persona: dict[str, Any] | None = None
    voice: str | None = None
    language: str | None = None
    llm_model: str | None = None
    user_id: str | None = None
    npc_id: str | None = None
    context: dict[str, Any] | None = None
    # Tenant + sessions.id UUIDs — used by KBRetrievalProcessor to call
    # the retriever with correct tenancy and to write `retrieval_traces`
    # rows that FK-link to the owning session. Strings here (not UUID
    # objects) so this dataclass stays cheap to construct / serialize;
    # the processor parses at its boundary. Both None → retriever is
    # disabled for the session (pass-through mode).
    tenant_id: str | None = None
    session_id: str | None = None
    # When set, selects the TTS engine for this session. Validated against
    # the same whitelist used by pipeline.AgentConfig; unknown or empty
    # values are dropped (fall back to cfg.tts_engine). This is how the
    # /demo site's three-button selector routes each session to a
    # different backend without restarting the agent.
    tts_engine: str | None = None

    @classmethod
    def from_dispatch(cls, body: dict[str, Any]) -> PerSessionOverrides:
        """Build from the raw /dispatch JSON body. Safe for arbitrary input."""

        def _as_str(value: Any) -> str | None:
            return value if isinstance(value, str) and value else None

        def _as_dict(value: Any) -> dict[str, Any] | None:
            return value if isinstance(value, dict) and value else None

        def _as_tts_engine(value: Any) -> str | None:
            # Drop unknown values silently rather than raising — the factory
            # still raises ValueError on unknown engine names, but the API
            # validates first so anything reaching here should already be in
            # the whitelist. Defense-in-depth: a misbehaving client can't
            # crash the agent by sending a bogus engine string.
            if isinstance(value, str) and value in _VALID_TTS_ENGINES:
                return value
            return None

        return cls(
            persona=_as_dict(body.get("persona")),
            voice=_as_str(body.get("voice")),
            language=_as_str(body.get("language")),
            llm_model=_as_str(body.get("llm_model")),
            user_id=_as_str(body.get("user_id")),
            npc_id=_as_str(body.get("npc_id")),
            context=_as_dict(body.get("context")),
            tts_engine=_as_tts_engine(body.get("tts_engine")),
            tenant_id=_as_str(body.get("tenant_id")),
            session_id=_as_str(body.get("session_id")),
        )

    @property
    def effective_language(self) -> str:
        """Language code, falling back to DEFAULT_LANGUAGE."""
        return self.language or DEFAULT_LANGUAGE


def build_system_prompt(
    persona: dict[str, Any] | None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Compose the LLM system prompt from persona + language.

    Precedence:
        1. persona.system_prompt — full override, used verbatim.
        2. name + backstory + style, then the language-specific voice rules.
        3. Just the voice rules when no persona is given.
    """
    if persona and isinstance(persona.get("system_prompt"), str):
        text = persona["system_prompt"].strip()
        if text:
            return text

    rules = VOICE_RULES_BY_LANGUAGE.get(language, VOICE_RULES_BY_LANGUAGE[DEFAULT_LANGUAGE])
    if not persona:
        return rules

    parts: list[str] = []
    name = persona.get("name") if isinstance(persona.get("name"), str) else None
    backstory = persona.get("backstory") if isinstance(persona.get("backstory"), str) else None
    style = persona.get("style") if isinstance(persona.get("style"), str) else None

    if name and backstory:
        parts.append(f"{name}. {backstory}")
    elif name:
        parts.append(name)
    elif backstory:
        parts.append(backstory)

    if style:
        parts.append(style)

    parts.append(rules)
    return " ".join(p.strip() for p in parts if p and p.strip())


def resolve_greeting(
    persona: dict[str, Any] | None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Pick the first-utterance greeting for this session."""
    if persona:
        custom = persona.get("greeting")
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    return DEFAULT_GREETINGS_BY_LANGUAGE.get(
        language, DEFAULT_GREETINGS_BY_LANGUAGE[DEFAULT_LANGUAGE]
    )

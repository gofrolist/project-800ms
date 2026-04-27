"""Runtime configuration for the retriever service.

Values come from environment variables (or `infra/.env` via docker compose
--env-file). Anything missing that the service genuinely needs MUST fail at
startup — we do not want a retriever silently running against a stub endpoint.
Constitution Principle IV / V: no hardcoded secrets, validate at boundary.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic Settings — all env-var wired."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── DB ──────────────────────────────────────────────────────────────
    # asyncpg DSN. Example:
    #   postgresql+asyncpg://voice:***@postgres:5432/voice
    db_url: str = Field(min_length=1)

    # ── Rewriter LLM (OpenAI-compatible endpoint shared with the agent) ──
    # AnyHttpUrl enforces scheme ∈ {http, https} at startup. A pasted
    # rogue-scheme or file:// URL would otherwise leak the Bearer token
    # to whatever the scheme resolved to on every /ready probe.
    llm_base_url: AnyHttpUrl
    llm_api_key: str = Field(min_length=1)
    rewriter_model: str = Field(min_length=1)
    # Hard timeout for the rewriter call; beyond this we fail-closed (refusal).
    rewriter_timeout_ms: int = Field(default=1500, ge=100, le=10_000)

    # ── Embedder ─────────────────────────────────────────────────────────
    # "cpu" | "cuda" | "cuda:0" — passed through to sentence-transformers.
    embedder_device: Literal["cpu", "cuda", "cuda:0", "cuda:1"] = "cpu"
    embedder_model: str = "BAAI/bge-m3"
    embedder_dim: int = 1024

    # ── Hybrid search ────────────────────────────────────────────────────
    hybrid_semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_lexical_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    default_top_k: int = Field(default=5, ge=1, le=20)

    # ── Service ──────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Internal auth ────────────────────────────────────────────────────
    # Shared-secret bearer for /retrieve. The agent presents it on every
    # call; the retriever rejects callers without it. Mirrors apps/api's
    # `agent_internal_token` pattern (header: X-Internal-Token).
    #
    # Empty in dev / tests is allowed but logs a warning at boot —
    # production deploys MUST set it (issue #47/#40). When empty the
    # endpoint accepts unauthenticated calls so existing tests +
    # one-shot eval harnesses keep working without ceremony.
    retriever_internal_token: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached via lru_cache so every module imports the same object without
    re-reading env vars. Tests may call `get_settings.cache_clear()` to
    force a reload when patching env vars.
    """
    return Settings()  # type: ignore[call-arg]

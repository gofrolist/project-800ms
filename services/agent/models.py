"""Shared model loading — heavy models are loaded once at startup.

GPU models (GigaAM) and CPU models (Piper, Silero VAD) are expensive to
initialize. This module loads them once and provides accessors for the
pipeline to use, so each dispatched room gets a near-instant pipeline
start instead of waiting 5-10s for model loading.

Startup fails hard if the loader raises — we do not want a degraded
agent serving traffic.
"""

from __future__ import annotations

import time

from loguru import logger


_gigaam_model: object | None = None


def load_gigaam(model_name: str = "v3_e2e_rnnt") -> object:
    """Load the GigaAM model once. Subsequent calls return the cached instance.

    Weights are fetched from HuggingFace on first call into HF_HOME
    (=/home/appuser/.cache/huggingface), which is mounted as the
    `hf_cache_agent` named volume in docker-compose.yml. First-run startup
    pays a one-time download (a few hundred MB — v3_e2e_rnnt carries the
    RNNT predictor/joint network plus the punctuation + text-normalization
    head on top of the shared v3 SSL backbone); subsequent container
    restarts hit the volume cache and load from local disk.

    Returns the gigaam.Model instance; typed as `object` because gigaam
    doesn't expose stable public type annotations.
    """
    global _gigaam_model
    if _gigaam_model is not None:
        return _gigaam_model

    # Local import so tests without gigaam installed (e.g. the agent's
    # existing test suite on a non-CUDA runner) can skip this without
    # hitting an ImportError at module scope.
    import gigaam  # noqa: PLC0415 — heavy import by design

    logger.info("Loading GigaAM model={model}", model=model_name)
    t0 = time.monotonic()
    _gigaam_model = gigaam.load_model(model_name)
    logger.info("GigaAM model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return _gigaam_model


def get_gigaam() -> object:
    """Return the pre-loaded GigaAM model. Raises if not loaded."""
    if _gigaam_model is None:
        raise RuntimeError("GigaAM model not loaded — call load_gigaam() first")
    return _gigaam_model


def _reset_models_for_tests() -> None:
    """Reset cached singletons. Tests only — do not call from runtime."""
    global _gigaam_model
    _gigaam_model = None

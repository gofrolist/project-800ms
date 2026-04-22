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
_silero_model: object | None = None


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


def load_silero(model_name: str = "v5_cis_base") -> object:
    """Load the Silero v5 Russian TTS model once. Cached on subsequent calls.

    Silero v5 has no PyPI package; the model is fetched via ``torch.hub``
    from ``snakers4/silero-models``. First call performs a network fetch
    into ``$TORCH_HOME/hub/`` (=/home/appuser/.cache/torch, mounted as
    the ``torch_cache_agent`` named volume in docker-compose.yml).
    Subsequent container restarts hit the volume cache and load from
    local disk without network.

    When a CUDA device is available we move the model to GPU after load
    — ``torch.hub.load`` returns a CPU-bound model by default and
    ``apply_tts`` would otherwise run on CPU even on the L4 host.

    Returns the Silero model instance; typed as ``object`` because
    ``torch.hub.load`` returns a ``torch.nn.Module`` whose exact type
    isn't re-exported from a stable public path.
    """
    global _silero_model
    if _silero_model is not None:
        return _silero_model

    # Local import — torch is a multi-hundred-MB dep and this module is
    # imported by the test suite on CPU-only CI where torch is not
    # present. Deferring the import matches the gigaam pattern above.
    import torch  # noqa: PLC0415 — heavy import by design

    logger.info("Loading Silero TTS model={model}", model=model_name)
    t0 = time.monotonic()
    # torch.hub.load returns (model, example_text_map) for silero_tts.
    # The example map is unused; discard it to avoid holding a reference
    # to test-corpus strings we'll never read.
    model, _example_text = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=model_name,
        trust_repo=True,
    )
    # Silero returns a CPU-bound model by default. Move to CUDA when a
    # device is present; fall back to CPU otherwise (CI, local dev on
    # macOS). apply_tts runs on whichever device the model is bound to.
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        logger.info("Silero model moved to CUDA")
    else:
        logger.info("Silero running on CPU (no CUDA device available)")
    _silero_model = model
    logger.info("Silero model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return _silero_model


def get_silero() -> object:
    """Return the pre-loaded Silero TTS model. Raises if not loaded."""
    if _silero_model is None:
        raise RuntimeError("Silero model not loaded — call load_silero() first")
    return _silero_model


def _reset_models_for_tests() -> None:
    """Reset cached singletons. Tests only — do not call from runtime."""
    global _gigaam_model, _silero_model
    _gigaam_model = None
    _silero_model = None

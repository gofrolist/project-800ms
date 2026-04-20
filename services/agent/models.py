"""Shared model loading — heavy models are loaded once at startup.

GPU models (Whisper, GigaAM) and CPU models (Piper, Silero VAD) are
expensive to initialize. This module loads them once and provides
accessors for the pipeline to use, so each dispatched room gets a
near-instant pipeline start instead of waiting 5-10s for model loading.

During the STT A/B experiment (see docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md)
both Whisper and GigaAM are loaded at startup so either can be selected
per-session via the `stack` dispatch field. Startup fails hard if either
loader raises — we do not want a degraded half-stack serving traffic.
"""

from __future__ import annotations

import time

from faster_whisper import WhisperModel
from loguru import logger


_whisper_model: WhisperModel | None = None
_gigaam_model: object | None = None


def load_whisper(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
) -> WhisperModel:
    """Load the Whisper model once. Subsequent calls return the cached instance."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    logger.info(
        "Loading Whisper model={model} device={device} compute={compute}",
        model=model_size,
        device=device,
        compute=compute_type,
    )
    t0 = time.monotonic()
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Whisper model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return _whisper_model


def get_whisper() -> WhisperModel:
    """Return the pre-loaded Whisper model. Raises if not loaded."""
    if _whisper_model is None:
        raise RuntimeError("Whisper model not loaded — call load_whisper() first")
    return _whisper_model


def load_gigaam(model_name: str = "v3_ctc") -> object:
    """Load the GigaAM model once. Subsequent calls return the cached instance.

    Weights are baked into the agent Docker image at build time (see the
    Dockerfile's `gigaam.load_model(...)` step), so at runtime this is a
    CPU→GPU transfer of already-local weights — no network I/O.

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
    global _whisper_model, _gigaam_model
    _whisper_model = None
    _gigaam_model = None

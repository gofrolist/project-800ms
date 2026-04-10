"""Shared model loading — heavy models are loaded once at startup.

GPU models (Whisper) and CPU models (Piper, Silero VAD) are expensive to
initialize. This module loads them once and provides accessors for the
pipeline to use, so each dispatched room gets a near-instant pipeline
start instead of waiting 5-10s for model loading.
"""

from __future__ import annotations

import time

from faster_whisper import WhisperModel
from loguru import logger


_whisper_model: WhisperModel | None = None


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

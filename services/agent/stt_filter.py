"""Whisper STT subclass with confidence-based hallucination filtering.

Instead of maintaining a blocklist of known hallucination strings, this
filters segments by their statistical properties:
  - no_speech_prob: high values indicate the segment is likely silence/noise
  - avg_logprob: low values indicate the model is uncertain about the text
  - compression_ratio: the magic value 0.5555… is a known hallucination marker

This catches hallucinations regardless of language or content.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings
from pipecat.utils.time import time_now_iso8601

# Compression ratio that faster-whisper produces for pure hallucinations
# (repeated tokens / padding). The MLX backend already checks this.
_HALLUCINATION_COMPRESSION_RATIO = 0.5555555555555556


@dataclass
class FilteredWhisperSettings(WhisperSTTSettings):
    """Extended settings with confidence thresholds."""

    # Segments with avg_logprob below this are dropped.
    # Real speech is typically > -0.5; hallucinations are often < -0.7.
    min_avg_logprob: float = -0.7

    # Segments with compression_ratio above this are likely repetitive
    # hallucinations (e.g. "ha ha ha ha ha").
    max_compression_ratio: float = 2.4


class FilteredWhisperSTTService(WhisperSTTService):
    """WhisperSTTService that drops low-confidence segments.

    Drop-in replacement — same constructor interface, just uses
    FilteredWhisperSettings for the extra thresholds. Accepts an optional
    pre-loaded WhisperModel to avoid reloading on each pipeline start.
    """

    Settings = FilteredWhisperSettings
    _settings: Settings

    def __init__(self, *, model: object | None = None, **kwargs):
        self._shared_model = model
        super().__init__(**kwargs)
        # Set after super().__init__ so it survives any init-time resets.
        if model is not None:
            self._model = model

    async def _load(self):
        """Use the pre-loaded model if available, otherwise load normally."""
        if self._shared_model is not None:
            self._model = self._shared_model
            return
        await super()._load()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[ErrorFrame | TranscriptionFrame, None]:
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = await asyncio.to_thread(
            self._model.transcribe, audio_float, language=self._settings.language
        )

        text = ""
        for seg in segments:
            # High probability that the segment is not speech at all
            if seg.no_speech_prob >= self._settings.no_speech_prob:
                logger.debug(
                    "Dropped segment (no_speech_prob={p:.2f}): {t}",
                    p=seg.no_speech_prob,
                    t=seg.text,
                )
                continue

            # Known hallucination compression ratio
            if abs(seg.compression_ratio - _HALLUCINATION_COMPRESSION_RATIO) < 1e-9:
                logger.debug(
                    "Dropped segment (compression_ratio={c}): {t}",
                    c=seg.compression_ratio,
                    t=seg.text,
                )
                continue

            # Very high compression = repetitive garbage
            if seg.compression_ratio > self._settings.max_compression_ratio:
                logger.debug(
                    "Dropped segment (compression_ratio={c:.2f}): {t}",
                    c=seg.compression_ratio,
                    t=seg.text,
                )
                continue

            # Low confidence = model is guessing
            if seg.avg_logprob < self._settings.min_avg_logprob:
                logger.debug(
                    "Dropped segment (avg_logprob={l:.2f}): {t}",
                    l=seg.avg_logprob,
                    t=seg.text,
                )
                continue

            text += f"{seg.text} "

        await self.stop_processing_metrics()

        if text:
            await self._handle_transcription(text, True, self._settings.language)
            logger.debug("Transcription: [{text}]", text=text)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._settings.language,
            )

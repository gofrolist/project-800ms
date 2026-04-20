"""Pipecat-compatible STT wrapper around GigaAM-v3.

Experiment half-life: this module exists for the STT A/B experiment
(docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md). If GigaAM
wins, this becomes the default STT and FilteredWhisperSTTService is
deleted; if Whisper wins, this file is deleted. Either way, it's
short-lived scaffolding — don't build infrastructure on top of it.

Base class choice — `SegmentedSTTService` not `WhisperSTTService`:
GigaAM has no Whisper-style per-segment confidence signals
(no_speech_prob, avg_logprob, compression_ratio), so subclassing the
Whisper base would force us to synthesize fake values. SegmentedSTTService
is the right abstraction — per-VAD-segment audio in, TranscriptionFrame
out — and matches how the current Pipecat pipeline feeds STT input.

Filter parity — pinned as an explicit second variable for the A/B:
FilteredWhisperSTTService drops segments by statistical signals;
GigaAM doesn't expose those signals, so we reject by behavioural
parity instead — same gross outcome ("short / near-empty utterances
are rejected"), different mechanism (duration + token gate instead of
logprob + compression). Threshold values will be tuned in Unit 3
against the WER eval set so both stacks reject a similar proportion
of false positives. The defaults below are the starting point; the
tuning outcome gets committed to `services/agent/eval/results/`.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from loguru import logger
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

# LiveKit publishes 16 kHz mono to the agent; Pipecat does the downsample
# upstream, so audio arriving here is always 16k mono int16.
_SAMPLE_RATE = 16_000

# Default filter thresholds — see module docstring. Tuned empirically in
# Unit 3 (WER smoke harness) against the committed Russian eval set.
_DEFAULT_MIN_DURATION_SECONDS = 0.30
_DEFAULT_MIN_TOKEN_COUNT = 2


@dataclass
class GigaAMSettings:
    """Per-instance settings for the GigaAM STT service.

    Kept lightweight — GigaAM exposes a `.transcribe(audio)` interface
    and little else; there's nothing to configure per-utterance.
    """

    language: str = "ru"
    model_name: str = "v3_ctc"  # alternatives: "v3_rnnt"
    min_duration_seconds: float = _DEFAULT_MIN_DURATION_SECONDS
    min_token_count: int = _DEFAULT_MIN_TOKEN_COUNT


class GigaAMSTTService(SegmentedSTTService):
    """Segmented STT service wrapping GigaAM-v3.

    Accepts an optional pre-loaded GigaAM model (same pattern as
    FilteredWhisperSTTService with its `model` kwarg) to avoid reloading
    on each pipeline start. Runtime agent preloads via
    `models.load_gigaam()` at startup.
    """

    def __init__(
        self,
        *,
        model: object | None = None,
        settings: GigaAMSettings | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gigaam_settings = settings or GigaAMSettings()
        # Bind the injected model eagerly so the service is usable the
        # moment it's constructed. Pipecat 0.0.108's SegmentedSTTService
        # does not call _load() on this path — deferring to _load() left
        # _loaded_model=None in prod and every voice segment errored
        # with "GigaAM model not available" (observed on the first live
        # deploy of the gigaam stack). Keep _load() around only as a
        # lazy fallback for callers that construct the service without
        # preloading a model.
        self._loaded_model: object | None = model

    async def _load(self):
        """Lazy-load a GigaAM model when one wasn't injected.

        Not reached in prod — main.py's load_gigaam() + get_gigaam()
        preload and inject the shared singleton before the first
        pipeline builds. Kept as a safety net for non-preloading
        callers and for parity with FilteredWhisperSTTService.
        """
        if self._loaded_model is not None:
            return
        # Local import matches the pattern in models.py — avoids hard
        # import of gigaam on non-GPU environments.
        import gigaam  # noqa: PLC0415

        self._loaded_model = await asyncio.to_thread(
            gigaam.load_model, self._gigaam_settings.model_name
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[ErrorFrame | TranscriptionFrame, None]:
        if self._loaded_model is None:
            # Shouldn't happen in prod — main.py preloads and injects the
            # singleton. Log the internal cause, emit a generic message to
            # the pipeline so client-visible errors don't disclose the stack.
            logger.error("GigaAM model not bound — preload/injection missing")
            yield ErrorFrame("STT unavailable")
            return

        await self.start_processing_metrics()
        try:
            # LiveKit + Pipecat upstream give us 16 kHz mono int16 PCM —
            # 2 bytes per sample, no header.
            sample_count = len(audio) // 2
            duration = sample_count / _SAMPLE_RATE

            # Filter 1 — duration gate. Whisper's filter drops short segments
            # via no_speech_prob; we approximate with a hard duration floor.
            if duration < self._gigaam_settings.min_duration_seconds:
                logger.debug(
                    "GigaAM dropped (duration={d:.3f}s < {m:.2f}s)",
                    d=duration,
                    m=self._gigaam_settings.min_duration_seconds,
                )
                return

            # GigaAM's transcribe() only accepts a file path — it shells out
            # to ffmpeg for decode. Write the incoming int16 PCM to a temp
            # WAV and pass the path. Overhead is ~1-2ms for the write plus
            # ~5-10ms for gigaam's internal ffmpeg subprocess. Acceptable
            # given we're comparing against faster-whisper's GPU path.
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="gigaam-stt-")
            os.close(fd)
            try:
                audio_int16 = np.frombuffer(audio, dtype=np.int16)
                sf.write(tmp_path, audio_int16, _SAMPLE_RATE, subtype="PCM_16")
                try:
                    result = await asyncio.to_thread(self._loaded_model.transcribe, tmp_path)
                except Exception:
                    # Keep exception detail in server logs only — the
                    # ErrorFrame can reach the LiveKit client and we don't
                    # want internal paths or gigaam internals leaking into
                    # a user-visible error surface.
                    logger.exception("GigaAM decode failed")
                    yield ErrorFrame("STT decode failed")
                    return
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # gigaam's transcribe() returns either a plain string or an
            # object with `.text`; normalise defensively since the library
            # doesn't publish a stable public return type.
            text = result.text.strip() if hasattr(result, "text") else str(result).strip()

            # Filter 2 — token count gate. Whisper's filter drops
            # repetitive hallucinations via compression_ratio; we approximate
            # with a minimum token count. Empty transcript (common for
            # silence segments GigaAM didn't drop itself) is covered by the
            # same check.
            tokens = text.split()
            if len(tokens) < self._gigaam_settings.min_token_count:
                logger.debug(
                    "GigaAM dropped (tokens={n} < {m}): {t}",
                    n=len(tokens),
                    m=self._gigaam_settings.min_token_count,
                    t=text,
                )
                return

            # FilteredWhisperSTTService calls self._handle_transcription() here
            # but it inherits from WhisperSTTService which provides that helper.
            # SegmentedSTTService (our base) does not expose it in Pipecat
            # 0.0.108, so calling it raises AttributeError and drops the frame.
            # The yielded TranscriptionFrame is the real data path; no helper
            # call is needed.
            logger.debug("GigaAM transcription: [{text}]", text=text)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._gigaam_settings.language,
            )
        finally:
            await self.stop_processing_metrics()

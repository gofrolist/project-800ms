"""Pipecat-compatible STT wrapper around GigaAM-v3.

Base class choice — `SegmentedSTTService` not `WhisperSTTService`:
GigaAM has no Whisper-style per-segment confidence signals
(no_speech_prob, avg_logprob, compression_ratio), so subclassing the
Whisper base would force us to synthesize fake values. SegmentedSTTService
is the right abstraction — per-VAD-segment audio in, TranscriptionFrame
out — and matches how the current Pipecat pipeline feeds STT input.

Filter rationale: the thresholds below (min duration + min token count)
reject short / near-empty utterances without leaning on statistical
confidence signals GigaAM doesn't expose. They were tuned during the
2026-04-19 STT A/B experiment against the committed Russian eval set.
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
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

# LiveKit publishes 16 kHz mono to the agent; Pipecat does the downsample
# upstream, so audio arriving here is always 16k mono int16.
_SAMPLE_RATE = 16_000

# Default filter thresholds — see module docstring. Tuned empirically in
# Unit 3 (WER smoke harness) against the committed Russian eval set.
_DEFAULT_MIN_DURATION_SECONDS = 0.30
_DEFAULT_MIN_TOKEN_COUNT = 2

# TTFS p99 measured via pipecat-ai/stt-benchmark on the GCP L4 host
# (2026-04-21, 40 samples from smart-turn-data-v3.1-train, vad_stop=0.3s):
#   p50=0.744s  p95=0.992s  p99=1.022s
# Pipecat's built-in 1.0s fallback assumes vad_stop=0.2; our 0.3s stop
# shifts the p99 slightly above. 1.05s absorbs the measurement with a
# small buffer against tail-samples not covered by 40 draws. Re-measure
# when model/hardware/VAD params change.
_MEASURED_TTFS_P99_LATENCY = 1.05


@dataclass
class GigaAMSettings:
    """Per-instance settings for the GigaAM STT service.

    Kept lightweight — GigaAM exposes a `.transcribe(audio)` interface
    and little else; there's nothing to configure per-utterance.
    """

    language: str = "ru"
    # v3_e2e_rnnt emits cased, punctuated, normalized text directly (no
    # post-processing step needed before feeding the LLM). RNNT over CTC
    # costs a small per-utterance latency delta for the autoregressive
    # predictor, but we already run non-streaming on VAD-bounded segments
    # so the streaming capability is unused either way — the choice here
    # is accuracy + formatting, not latency behaviour.
    # Alternatives: "v3_e2e_ctc" (same formatting, slightly faster, slightly
    # lower accuracy), "v3_ctc" / "v3_rnnt" (raw lowercase, no punctuation).
    model_name: str = "v3_e2e_rnnt"
    min_duration_seconds: float = _DEFAULT_MIN_DURATION_SECONDS
    min_token_count: int = _DEFAULT_MIN_TOKEN_COUNT


class GigaAMSTTService(SegmentedSTTService):
    """Segmented STT service wrapping GigaAM-v3.

    Accepts an optional pre-loaded GigaAM model to avoid reloading on each
    pipeline start. Runtime agent preloads via `models.load_gigaam()` at
    startup.
    """

    def __init__(
        self,
        *,
        gigaam_model: object | None = None,
        settings: GigaAMSettings | None = None,
        **kwargs,
    ):
        gigaam_settings = settings or GigaAMSettings()
        # Apply the benchmarked p99 default unless the caller overrode it.
        # Pipecat's base class hardcodes 1.0s when unset, which is calibrated
        # for vad_stop_secs=0.2 — our pipeline uses 0.3s.
        kwargs.setdefault("ttfs_p99_latency", _MEASURED_TTFS_P99_LATENCY)
        # Populate Pipecat's base STTSettings with the model name + language
        # so the framework's settings tracking has non-NOT_GIVEN values.
        # Without this, Pipecat logs:
        #   "STTSettings: the following fields are NOT_GIVEN: model, language"
        # on every pipeline start.
        super().__init__(
            settings=STTSettings(
                model=gigaam_settings.model_name,
                language=gigaam_settings.language,
            ),
            **kwargs,
        )
        self._gigaam_settings = gigaam_settings
        # Bind the injected model eagerly so the service is usable the
        # moment it's constructed. Pipecat 0.0.108's SegmentedSTTService
        # does not call _load() on this path — deferring to _load() left
        # _loaded_model=None in prod and every voice segment errored
        # with "GigaAM model not available" (observed on the first live
        # deploy of the gigaam stack). Keep _load() around only as a
        # lazy fallback for callers that construct the service without
        # preloading a model.
        self._loaded_model: object | None = gigaam_model

    async def _load(self):
        """Lazy-load a GigaAM model when one wasn't injected.

        Not reached in prod — main.py's load_gigaam() + get_gigaam()
        preload and inject the shared singleton before the first
        pipeline builds. Kept as a safety net for non-preloading callers.
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

            # SegmentedSTTService (our base) does not expose
            # _handle_transcription() in Pipecat 0.0.108 — the yielded
            # TranscriptionFrame is the data path; no helper call is needed.
            logger.debug("GigaAM transcription: [{text}]", text=text)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._gigaam_settings.language,
            )
        finally:
            await self.stop_processing_metrics()

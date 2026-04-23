"""Pipecat-compatible TTS wrapper around Silero v5.

Silero v5 has no PyPI package: the model is loaded via ``torch.hub`` and
exposes a single ``apply_tts(text, sample_rate, ...)`` method that returns
a float32 torch tensor of the whole utterance. This adapter wraps that
file-at-a-time synth into the Pipecat ``TTSService`` async-generator
contract.

Integration shape mirrors ``PiperTTSService`` (see
``services/agent/.venv/lib/python3.12/site-packages/pipecat/services/piper/tts.py``):

- ``push_start_frame=True`` + ``push_stop_frames=True`` so downstream
  transport layers see the TTS lifecycle and barge-in gating works.
- ``super().__init__(settings=TTSSettings(...), ...)`` to populate the
  framework's settings store; without this Pipecat logs a
  ``TTSSettings: the following fields are NOT_GIVEN`` warning on every
  pipeline start (same issue documented in ``gigaam_stt.py:93-104``).
- Eager model binding in ``__init__`` rather than a lazy ``_load()``:
  Pipecat 0.0.108's ``TTSService`` path does not call ``_load()``, so a
  deferred load would leave ``_loaded_model=None`` in prod (same
  observation as ``gigaam_stt.py:106-113``).

Barge-in trade-off — Silero returns the entire utterance before the first
frame yields, so an ``InterruptionFrame`` arriving mid-synth cannot
truncate the GPU work in progress (``asyncio.to_thread`` is
uncancellable). See
``docs/solutions/tts-selection/silero-spike-findings.md`` for the full
rationale; this is an accepted MVP limitation, not a bug.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from tts_utils import single_shot_audio_iterator

# Silero v5 ``v5_cis_base`` emits 24 kHz natively (also supports 8 kHz and
# 48 kHz). 24 kHz matches LiveKit's transport expectation, so there's no
# resample overhead when piping through
# ``_stream_audio_frames_from_iterator``. Changing this would break that
# contract silently — update ``sample_rate`` default below in lockstep.
_SILERO_SAMPLE_RATE = 24_000

# int16 PCM conversion factor. Silero returns float32 tensors in the range
# [-1, 1]; multiplying by 32767 (max int16) and casting to int16 yields
# LiveKit-compatible little-endian PCM bytes.
_INT16_SCALE = 32767

# Module-level lock that serializes ``apply_tts`` calls across every
# ``SileroTTSService`` instance in the process.
#
# The cached Silero model is a single ``torch.nn.Module`` stored in
# ``models._silero_cached``; every ``SileroTTSService`` created for a
# dispatched room holds a reference to the same Module. PyTorch's
# ``nn.Module`` is NOT thread-safe — concurrent ``apply_tts`` calls
# interleave writes to the model's hidden state via the default
# executor's worker threads and can corrupt tensors or raise obscure
# CUDA errors.
#
# Tradeoff: one TTS synth at a time, globally. At the experiment's
# single-session load this is invisible; the lock is held for the
# duration of a ``to_thread`` wrapped ``apply_tts`` call (tens of ms
# on L4). If we later scale to concurrent sessions, options are
# (a) clone the model per worker (expensive — ~200 MB each, and
# torch.hub doesn't give us a cheap deep-copy path for Silero's
# custom scripted module), or (b) move to a multi-GPU setup. For now,
# lock-serialize.
_silero_apply_lock = asyncio.Lock()


@dataclass
class SileroSettings:
    """Per-instance settings for the Silero TTS adapter.

    Minimal — Silero v5 Russian models are single-speaker, so the only
    knob that matters is which model to load (e.g. ``v5_cis_base``,
    ``v5_ru``). Accent + ё normalization are always on: they're cheap,
    they improve quality for the kind of game-domain text this pipeline
    handles, and there's no observed reason to expose them as tunables.

    Passthrough semantics: the fields here are consumed eagerly at
    ``__init__`` time — the speaker is echoed into ``TTSSettings.model``
    / ``TTSSettings.voice`` and the language into ``TTSSettings.language``.
    After construction there is no stored reference to the
    ``SileroSettings`` instance; it exists for API symmetry with the
    ``GigaAMSettings`` constructor pattern so operators wiring the
    service directly can opt into explicit configuration.
    """

    # Silero per-call speaker id. The v5_cis_base model bundles ~60
    # speakers across CIS languages — apply_tts(speaker=...) picks one
    # per synth call. Russian speakers are prefixed ``ru_`` (e.g.
    # ``ru_dmitriy``, ``ru_ekaterina``, ``ru_eduard``). Default matches
    # the model's own apply_tts default so a blank config keeps the
    # pre-existing voice. Passed into apply_tts at runtime; echoed into
    # TTSSettings.voice for framework visibility. DO NOT confuse with
    # the torch.hub.load ``speaker=`` arg used at MODEL load time
    # (that's always "v5_cis_base" for us — see models.load_silero).
    speaker: str = "ru_zhadyra"
    # Language code in the TTS settings store. Silero's Russian v5 model
    # only produces Russian output; attempting to synthesize other
    # languages will emit gibberish. The factory does not currently
    # route non-Russian traffic to this service.
    language: str = "ru"


class SileroTTSService(TTSService):
    """TTS service wrapping a preloaded Silero v5 model.

    The caller is expected to inject a model preloaded via
    ``models.load_silero()`` (runtime agent does this at startup when
    ``TTS_ENGINE=silero``). Passing ``silero_model=None`` is a
    degraded-mode path that logs and emits a redacted ``ErrorFrame`` on
    every synth attempt — mirrors ``GigaAMSTTService``'s behaviour when
    its model injection is missing.
    """

    def __init__(
        self,
        *,
        silero_model: object | None = None,
        settings: SileroSettings | None = None,
        **kwargs,
    ):
        silero_settings = settings or SileroSettings()
        # Match PiperTTSService precedent (piper/tts.py:94-95). Without
        # push_start_frame / push_stop_frames the downstream LiveKit
        # transport doesn't see TTS lifecycle frames and interruption
        # gating (VADUserStartedSpeakingFrame -> InterruptionFrame) can't
        # correctly truncate in-flight playback.
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        kwargs.setdefault("sample_rate", _SILERO_SAMPLE_RATE)
        # Populate Pipecat's TTSSettings with concrete values so the
        # framework's settings-tracking doesn't log a NOT_GIVEN warning on
        # every pipeline start. ``voice`` is the Silero speaker id so
        # runtime TTSUpdateSettingsFrame(voice=...) updates route through
        # the canonical path even though Silero can't actually swap
        # voices mid-stream (single-speaker model).
        super().__init__(
            settings=TTSSettings(
                model=silero_settings.speaker,
                voice=silero_settings.speaker,
                language=silero_settings.language,
            ),
            **kwargs,
        )
        # Bind eagerly. Pipecat 0.0.108's TTSService does not call
        # _load(); waiting for lazy load leaves _loaded_model=None in
        # prod and every synth errors with "TTS model not bound".
        # See gigaam_stt.py:106-113 for the same issue in the STT
        # subclass.
        self._loaded_model: object | None = silero_model

    def can_generate_metrics(self) -> bool:
        """Enable Pipecat's TTS usage + TTFB metrics for this service.

        Match ``PiperTTSService`` (piper/tts.py:116-122) — metrics are
        cheap and the bench harness in Unit 5 will read them back when
        computing per-engine TTFB + RTF.
        """
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` via Silero and yield Pipecat audio frames.

        File-at-a-time: the whole utterance is rendered off-thread before
        any frame yields. ``_stream_audio_frames_from_iterator`` then
        chunks the resulting bytes into ``TTSAudioRawFrame``\\s at the
        service's configured sample rate.
        """
        if self._loaded_model is None:
            # Same defensive path as gigaam_stt.py:134-140 — log the
            # internal cause, emit a redacted message to the pipeline.
            logger.error("Silero model not bound — preload/injection missing")
            # Stop the TTFB timer Pipecat started in _push_tts_frames;
            # every run_tts exit path must stop it exactly once or the
            # timer stays stuck.
            await self.stop_ttfb_metrics()
            yield ErrorFrame("TTS unavailable")
            return

        # Pipecat strips whitespace-only text upstream in _push_tts_frames
        # (tts_service.py:973-974), so a whitespace-only text shouldn't
        # reach us. A fully empty string, however, would make Silero's
        # apply_tts raise on some checkpoints — early-return with no
        # frame, matching the GigaAM precedent of dropping near-empty
        # segments quietly (gigaam_stt.py:151-157).
        if not text or not text.strip():
            logger.debug("Silero dropped empty text")
            await self.stop_ttfb_metrics()  # see TTFB note on the guard above
            return

        logger.debug("Silero TTS: synthesizing [{text}]", text=text)
        # Sentinel flag so stop_ttfb_metrics() fires exactly once per
        # synth — on the first yielded frame on the happy path, or in
        # ``finally`` on the error path before any frame has flowed.
        # Previously it was called inside the async-for loop AND in
        # ``finally``, so a single happy-path synth incremented the TTFB
        # stop counter twice.
        ttfb_stopped = False
        try:
            await self.start_tts_usage_metrics(text)

            # Blocking synth off-thread. On L4 GPU this takes ~60ms for
            # a one-second utterance (RTF ~0.06 per upstream bench).
            # The module-level lock serializes ``apply_tts`` so two
            # concurrent ``SileroTTSService`` instances can't corrupt
            # the shared ``nn.Module``'s hidden state via interleaved
            # threadpool workers. See ``_silero_apply_lock`` comment.
            async with _silero_apply_lock:
                audio_tensor = await asyncio.to_thread(
                    self._loaded_model.apply_tts,
                    text=text,
                    speaker=self._settings.voice,
                    sample_rate=_SILERO_SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True,
                )

            # Silero returns float32 in [-1, 1]; LiveKit expects int16
            # PCM little-endian. 32767 (not 32768) keeps the positive
            # saturation one step below the int16 max — matches the
            # standard WAV/CD conversion and avoids a rare one-sample
            # overflow when the model clips to exactly +1.0.
            # ``.detach().cpu().numpy()`` handles both CPU- and
            # CUDA-resident tensors: ``.numpy()`` alone raises
            # "can't convert cuda tensor" when the model is on GPU, and
            # ``.detach()`` drops any autograd state just in case.
            audio_np = audio_tensor.detach().cpu().numpy()
            pcm_bytes = (audio_np * _INT16_SCALE).astype(np.int16).tobytes()

            async for frame in self._stream_audio_frames_from_iterator(
                single_shot_audio_iterator(pcm_bytes),
                in_sample_rate=_SILERO_SAMPLE_RATE,
                context_id=context_id,
            ):
                # Stop the TTFB timer on the first yielded frame — before
                # the frame is consumed downstream. Matches PiperTTSService
                # (piper/tts.py:171). Only fires once; the sentinel flag
                # guards against duplicate calls from later iterations +
                # the ``finally`` block below.
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True
                yield frame
        except Exception:
            # Keep exception detail in server logs only — the ErrorFrame
            # is visible to the LiveKit client and must not leak internal
            # state (stack traces, model internals, CUDA error codes).
            # Mirrors gigaam_stt.py:171-177.
            logger.exception("Silero TTS synth failed")
            yield ErrorFrame("TTS synth failed")
        finally:
            # Only stop TTFB here if the happy path never yielded a
            # frame — i.e. the error path fired before the async-for
            # produced anything. Otherwise the async-for block already
            # called it exactly once.
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            logger.debug("Silero TTS: finished [{text}]", text=text)

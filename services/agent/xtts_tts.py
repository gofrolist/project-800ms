"""Pipecat-compatible TTS wrapper around Coqui XTTS v2.

XTTS v2 is a multilingual (17-language) zero-shot voice-cloning TTS model
distributed via the maintained ``coqui-tts`` PyPI package. The weights
themselves are released under the Coqui Public Model License (non-
commercial) — fine for a demo, but callers shipping to paying customers
need to verify their licensing posture before enabling this engine.

Integration shape mirrors ``SileroTTSService``:

* File-at-a-time synth via ``TTS.api.TTS.tts()`` off-thread through
  ``asyncio.to_thread``. ``_stream_audio_frames_from_iterator`` chunks the
  PCM bytes into ``TTSAudioRawFrame``\\s at the negotiated 24 kHz.
* Eager model bind in ``__init__`` — Pipecat 0.0.108's ``TTSService``
  path does not call ``_load()``, so a deferred load would leave
  ``_loaded_model=None`` in prod and every synth would ErrorFrame.
* Module-level ``asyncio.Lock`` serializes ``tts()`` across instances —
  the underlying model is a torch ``nn.Module`` and is NOT thread-safe.
  Same tradeoff as Silero: one synth at a time, globally. Cloning the
  XTTS model per worker would cost ~1.8 GB of VRAM apiece.

Voice-library integration — unlike Piper/Silero which take a string
speaker id, XTTS v2 is a zero-shot voice cloner: every synth needs
reference audio (~6-30 s of the target voice) plus a language code.
This adapter reuses the ``voice_library/profiles/<profile>/`` directory
shape used by the Qwen3 sidecar so operators can swap a single reference
clip and have both engines pick it up. The voice string coming through
``build_tts_service`` is expected to be in the ``clone:<profile>`` form
(e.g. ``clone:demo-ru``); the adapter resolves that at construction
time to a concrete ``ref.wav`` path and an XTTS language code.

Barge-in trade-off — XTTS's high-level ``.tts()`` returns the entire
utterance before the first frame yields, so an ``InterruptionFrame``
arriving mid-synth cannot truncate the GPU work in progress
(``asyncio.to_thread`` is uncancellable). Same accepted MVP limitation
as Silero — see ``docs/solutions/tts-selection/silero-spike-findings.md``.
A follow-up to the low-level ``Xtts.inference_stream()`` API would give
us true streaming (first chunk ~200 ms on L4) and true interruption,
but doubles the code complexity; defer until the demo shows a real need.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from tts_utils import single_shot_audio_iterator

# XTTS v2 emits 24 kHz natively (``tts_models/multilingual/multi-dataset/
# xtts_v2`` output_sample_rate in the published config). 24 kHz matches
# LiveKit's transport expectation, so ``_stream_audio_frames_from_iterator``
# does no resample work. Changing this would break that contract silently.
_XTTS_SAMPLE_RATE = 24_000

# int16 PCM conversion factor. XTTS returns float32 in [-1, 1]; 32767 is
# the int16 max with one step headroom against the rare +1.0 sample
# (matches Silero + WAV/CD convention).
_INT16_SCALE = 32767

# XTTS v2 supports 17 languages — the codes below are the canonical
# forms accepted by the ``tts_models/multilingual/multi-dataset/xtts_v2``
# config. Language lookup in ``_resolve_voice_profile`` is two-step:
#   1. If the meta.json value is a human-readable alias (``"Russian"``
#      → ``"russian"`` via str.lower().strip()), translate via
#      ``_XTTS_LANGUAGE_NAME_TO_CODE``.
#   2. Otherwise, if the value is already one of the 17 codes, pass
#      through unchanged.
#   3. Anything else falls back to ``_DEFAULT_XTTS_LANGUAGE``.
# Splitting alias-vs-code-set this way keeps each language a single
# entry in the alias table and lets the valid-code set be asserted
# independently — the previous flat dict duplicated every code as its
# own self-entry, making "adding a new language" a two-edit operation.
_XTTS_LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "arabic": "ar",
    "chinese": "zh-cn",
    "zh": "zh-cn",
    "czech": "cs",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
}

# The 17-language canonical code set XTTS v2 accepts. Kept as a frozenset
# (not derived from the alias dict values) so adding a new alias requires
# confirming the target code is actually supported.
_XTTS_VALID_CODES: frozenset[str] = frozenset(
    {
        "ar",
        "cs",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "hu",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "tr",
        "zh-cn",
    }
)

# Primary language of this demo. Used when the profile's meta.json lacks
# a ``language`` field or carries an unknown value. Picking the wrong
# language silently butchers prosody but doesn't crash the service.
_DEFAULT_XTTS_LANGUAGE = "ru"

# Module-level lock that serializes ``tts()`` calls across every
# ``XTTSTTSService`` instance in the process. Same rationale as Silero's
# ``_silero_apply_lock``: PyTorch ``nn.Module`` is NOT thread-safe and
# two concurrent synths on the shared model would interleave hidden
# state through the threadpool executor's workers. Holding the lock for
# a full synth (~1-3 s on L4) is acceptable at the MVP's single-session
# load; if we scale concurrency, the options are (a) per-worker model
# clones (~1.8 GB VRAM each) or (b) multi-GPU deploys.
_xtts_apply_lock = asyncio.Lock()


@dataclass(frozen=True)
class XTTSSettings:
    """Per-instance settings for the XTTS TTS adapter.

    ``voice`` must be in the ``clone:<profile>`` form — XTTS is a zero-shot
    voice cloner and cannot synthesize from an abstract speaker id.
    ``voice_library_dir`` points at the root of a voice-library mount
    (see ``voice_library/README.md`` for directory shape); the adapter
    resolves ``clone:<profile>`` against ``<dir>/profiles/<profile>/``
    at service construction time, not per synth call.

    Passthrough semantics: fields are consumed eagerly in ``__init__``
    (no stored reference to the ``XTTSSettings`` instance after init).
    """

    voice: str
    voice_library_dir: Path


@dataclass(frozen=True)
class _ResolvedProfile:
    """Concrete profile-on-disk derived from a ``clone:<profile>`` voice id."""

    profile_id: str
    ref_audio_path: Path
    language: str  # XTTS language code (e.g. "ru")


def _resolve_voice_profile(
    voice: str,
    *,
    voice_library_dir: Path,
) -> _ResolvedProfile:
    """Parse ``clone:<profile>`` into a concrete ref audio path + language code.

    Raises ``ValueError`` on missing/unreadable/malformed profiles so the
    failure surfaces at service construction time (which pipeline.build_task
    runs on dispatch) rather than at the first synth attempt — a bad profile
    would otherwise play silence and emit an ``ErrorFrame`` only after the
    user had already spoken.
    """
    if not voice.startswith("clone:"):
        raise ValueError(
            f"XTTS requires voice in 'clone:<profile>' form; got {voice!r}. "
            f"See voice_library/README.md for the profile directory shape."
        )
    profile_id = voice[len("clone:") :]
    if not profile_id:
        raise ValueError("XTTS voice profile id is empty after 'clone:' prefix")

    profile_dir = voice_library_dir / "profiles" / profile_id
    meta_path = profile_dir / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"XTTS voice profile meta.json not found: {meta_path}")

    try:
        meta_text = meta_path.read_text(encoding="utf-8")
        meta = json.loads(meta_text)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read XTTS voice profile meta {meta_path}: {e}") from e

    if not isinstance(meta, dict):
        raise ValueError(f"XTTS voice profile meta {meta_path} is not a JSON object")

    ref_filename = meta.get("ref_audio_filename", "ref.wav")
    if not isinstance(ref_filename, str) or not ref_filename:
        raise ValueError(
            f"XTTS voice profile meta {meta_path} ref_audio_filename is missing or empty"
        )
    ref_audio_path = profile_dir / ref_filename
    if not ref_audio_path.is_file():
        # Include the meta.json path so operators who forgot to set
        # ``ref_audio_filename`` (implicit default "ref.wav") can see
        # where to update vs. a genuinely missing audio file.
        raise ValueError(
            f"XTTS ref audio not found: {ref_audio_path}. "
            f"Check 'ref_audio_filename' in {meta_path}."
        )

    language_raw = meta.get("language", "")
    language_key = str(language_raw).lower().strip()
    # Two-step resolution: alias → code → default. See the
    # _XTTS_LANGUAGE_NAME_TO_CODE / _XTTS_VALID_CODES split above.
    language_code = _XTTS_LANGUAGE_NAME_TO_CODE.get(
        language_key,
        language_key if language_key in _XTTS_VALID_CODES else _DEFAULT_XTTS_LANGUAGE,
    )

    return _ResolvedProfile(
        profile_id=profile_id,
        ref_audio_path=ref_audio_path,
        language=language_code,
    )


class XTTSTTSService(TTSService):
    """Pipecat TTS service wrapping a preloaded Coqui XTTS v2 model.

    The caller is expected to inject a model preloaded via
    ``models.load_xtts()`` (runtime agent does this at startup when
    ``xtts`` appears in ``TTS_PRELOAD_ENGINES``). Passing
    ``xtts_model=None`` is a degraded-mode path that logs and emits a
    redacted ``ErrorFrame`` on every synth attempt — mirrors
    ``GigaAMSTTService`` / ``SileroTTSService`` behavior when their model
    injection is missing.
    """

    def __init__(
        self,
        *,
        xtts_model: object | None = None,
        settings: XTTSSettings,
        **kwargs: Any,
    ):
        # Resolve the voice profile eagerly so a bad profile identifier
        # or missing ref audio fails at service construction time, not on
        # the first synth attempt when the user has already spoken and
        # is waiting for a response. The ValueError propagates out of
        # ``__init__`` — pipeline.build_task runs inside the dispatch
        # handler's try/except and will mark the room failed.
        resolved = _resolve_voice_profile(
            settings.voice,
            voice_library_dir=settings.voice_library_dir,
        )

        # Match PiperTTSService / SileroTTSService precedent: without
        # push_start_frame + push_stop_frames the downstream LiveKit
        # transport doesn't see the TTS lifecycle and interruption
        # gating (VADUserStartedSpeakingFrame → InterruptionFrame) can't
        # correctly truncate in-flight playback.
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        kwargs.setdefault("sample_rate", _XTTS_SAMPLE_RATE)
        super().__init__(
            settings=TTSSettings(
                model="xtts_v2",
                voice=settings.voice,
                language=resolved.language,
            ),
            **kwargs,
        )
        # Bind eagerly. Pipecat 0.0.108's TTSService does not call
        # _load(); waiting for lazy load leaves _loaded_model=None in
        # prod and every synth errors with "TTS unavailable". Same
        # observation as gigaam_stt.py + silero_tts.py.
        self._loaded_model: object | None = xtts_model
        self._ref_audio_path = resolved.ref_audio_path
        self._language = resolved.language
        self._profile_id = resolved.profile_id

    def can_generate_metrics(self) -> bool:
        """Enable Pipecat's TTS usage + TTFB metrics for this service."""
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` via XTTS v2 and yield Pipecat audio frames.

        File-at-a-time: the whole utterance is rendered off-thread before
        any frame yields. ``_stream_audio_frames_from_iterator`` then
        chunks the resulting PCM bytes into ``TTSAudioRawFrame``\\s at
        the negotiated 24 kHz.
        """
        if self._loaded_model is None:
            logger.error("XTTS model not bound — preload/injection missing")
            yield ErrorFrame("TTS unavailable")
            return

        # Pipecat strips whitespace-only text upstream in _push_tts_frames
        # (tts_service.py:973-974), so whitespace-only shouldn't reach us.
        # An empty string would make XTTS's tokenizer raise on some code
        # paths — early-return silently, matching Silero's precedent.
        if not text or not text.strip():
            logger.debug("XTTS dropped empty text")
            return

        logger.debug(
            "XTTS: synthesizing profile={profile} text=[{text}]",
            profile=self._profile_id,
            text=text,
        )
        # Sentinel flag so stop_ttfb_metrics() fires exactly once per
        # synth — on the first yielded frame on the happy path, or in
        # ``finally`` on the error path before any frame has flowed.
        # Same pattern as silero_tts.py.
        ttfb_stopped = False
        try:
            await self.start_tts_usage_metrics(text)

            # Blocking synth off-thread. On L4 GPU a ~5 s utterance
            # takes ~1.5-2 s (RTF ~0.3-0.4). The module-level lock
            # serializes ``tts()`` across concurrent instances so
            # interleaved threadpool workers can't corrupt the shared
            # nn.Module's hidden state. See ``_xtts_apply_lock`` comment.
            async with _xtts_apply_lock:
                wav_out = await asyncio.to_thread(
                    self._loaded_model.tts,
                    text=text,
                    speaker_wav=str(self._ref_audio_path),
                    language=self._language,
                )

            # ``TTS.api.TTS.tts()`` returns ``list[float]``, ``np.ndarray``,
            # or — on GPU synth paths — a CUDA ``torch.Tensor``.
            # ``np.asarray`` on a CUDA tensor raises ``TypeError: can't
            # convert cuda tensor to numpy. Use .cpu() first`` which the
            # broad ``except`` below would catch and redact as an opaque
            # ErrorFrame. Duck-type on ``.detach`` (present on every
            # ``torch.Tensor`` subclass, absent on list/ndarray) to route
            # tensors through the correct conversion chain.
            if hasattr(wav_out, "detach"):
                wav_out = wav_out.detach().cpu().numpy()
            audio_np = np.asarray(wav_out, dtype=np.float32)
            pcm_bytes = (audio_np * _INT16_SCALE).astype(np.int16).tobytes()

            async for frame in self._stream_audio_frames_from_iterator(
                single_shot_audio_iterator(pcm_bytes),
                in_sample_rate=_XTTS_SAMPLE_RATE,
                context_id=context_id,
            ):
                # Stop the TTFB timer on the first yielded frame — before
                # the frame is consumed downstream. Matches PiperTTSService
                # (piper/tts.py:171) and SileroTTSService's pattern.
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True
                yield frame
        except Exception:
            # Keep exception detail in server logs only — the ErrorFrame
            # is visible to the LiveKit client and must not leak internal
            # state (stack traces, CUDA error codes, file paths).
            logger.exception("XTTS synth failed")
            yield ErrorFrame("TTS synth failed")
        finally:
            # Only stop TTFB here if the happy path never yielded a
            # frame — i.e. the error path fired before the async-for
            # produced anything.
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            logger.debug(
                "XTTS: finished profile={profile} text=[{text}]",
                profile=self._profile_id,
                text=text,
            )

"""Tests for SileroTTSService + models.load_silero().

torch IS installed in the agent venv (gigaam[torch] pulls it), but
``torch.hub.load`` would perform network IO against
``snakers4/silero-models`` on every invocation. We monkey-patch
``torch.hub.load`` + ``torch.cuda.is_available`` per-test to control the
returned model object without touching the network or requiring an
actual CUDA device.

gigaam is not available in the CPU-only CI environment — stub it at
module scope the same way test_gigaam_stt.py does so shared imports
through ``models`` don't fail when this test module happens to load
first.

Convention: tests are sync and use ``asyncio.run()`` for async bodies,
matching the existing test_gigaam_stt.py / test_transcript_sink.py
pattern. The agent test suite deliberately avoids pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# gigaam lives alongside silero in models.py; stub it so shared imports
# through ``models`` don't fail when this test module loads first.
sys.modules.setdefault("gigaam", MagicMock())

from models import (  # noqa: E402 — must import after stub injection
    _reset_models_for_tests,
    get_silero,
    load_silero,
)
from silero_tts import (  # noqa: E402
    SileroSettings,
    SileroTTSService,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_models_for_tests()
    yield
    _reset_models_for_tests()


@pytest.fixture
def mock_torch_hub(monkeypatch):
    """Patch torch.hub.load + torch.cuda.is_available for a test.

    Returns a SimpleNamespace with ``hub_load`` (MagicMock) and a helper
    to toggle CUDA availability. Keeps the real torch module intact so
    other pipecat imports that reflect on torch keep working.
    """
    import torch  # noqa: PLC0415

    hub_load_mock = MagicMock(name="torch.hub.load")
    cuda_available_mock = MagicMock(name="torch.cuda.is_available", return_value=False)
    monkeypatch.setattr(torch.hub, "load", hub_load_mock)
    monkeypatch.setattr(torch.cuda, "is_available", cuda_available_mock)
    return _TorchHubFixture(hub_load=hub_load_mock, cuda_available=cuda_available_mock)


class _TorchHubFixture:
    def __init__(self, *, hub_load, cuda_available):
        self.hub_load = hub_load
        self.cuda_available = cuda_available


async def _drain(async_gen):
    """Collect frames from an AsyncGenerator into a list."""
    return [frame async for frame in async_gen]


def _make_audio_tensor(n_samples: int = 24000) -> MagicMock:
    """Build a mock torch tensor with a ``numpy()`` method.

    Silero returns a torch.Tensor whose ``.numpy()`` method gives a
    float32 ndarray in [-1, 1]. We mock the shape: a real tensor with
    non-zero content flows through the int16 conversion without the
    whole array collapsing to zero bytes.
    """
    tensor = MagicMock(name="audio_tensor")
    # Use a small non-zero signal so the int16 conversion produces
    # non-empty audio bytes the framing helper doesn't drop.
    tensor.numpy.return_value = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
    return tensor


def _build_service(*, apply_tts_return, settings=None):
    """Factory for a SileroTTSService backed by a mock silero model.

    Overrides the Pipecat metrics hooks with AsyncMock so tests don't
    have to stand up the full service runtime, and sets
    ``_sample_rate`` directly — in production this is set by
    ``TTSService.start()`` from the StartFrame, but we're not
    exercising the full pipeline lifecycle here.
    """
    fake_model = MagicMock(name="silero_model")
    fake_model.apply_tts = MagicMock(return_value=apply_tts_return)
    svc = SileroTTSService(silero_model=fake_model, settings=settings)
    svc._loaded_model = fake_model
    # These would normally be called via the base class's frame
    # processing plumbing — short-circuit them so run_tts is testable
    # in isolation.
    svc.start_tts_usage_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    # sample_rate defaults to 0 until start() is called — set it
    # directly to the value __init__ stashed in _init_sample_rate so
    # _stream_audio_frames_from_iterator produces correctly-framed
    # TTSAudioRawFrames.
    svc._sample_rate = svc._init_sample_rate or 24000
    return svc, fake_model


# ─── models.load_silero / get_silero ─────────────────────────────────


class TestLoadSilero:
    def test_loads_once_and_caches(self, mock_torch_hub):
        fake_model = MagicMock(name="silero_model")
        mock_torch_hub.hub_load.return_value = (fake_model, {"example": "text"})

        first = load_silero("v5_cis_base")
        second = load_silero("v5_cis_base")

        assert first is fake_model
        assert second is fake_model
        # Cached singleton — torch.hub.load called exactly once across
        # two load_silero() invocations.
        assert mock_torch_hub.hub_load.call_count == 1

    def test_load_passes_speaker_and_repo_to_torch_hub(self, mock_torch_hub):
        fake_model = MagicMock(name="silero_model")
        mock_torch_hub.hub_load.return_value = (fake_model, {})

        load_silero("v5_cis_base")

        kwargs = mock_torch_hub.hub_load.call_args.kwargs
        assert kwargs["repo_or_dir"] == "snakers4/silero-models"
        assert kwargs["model"] == "silero_tts"
        assert kwargs["language"] == "ru"
        assert kwargs["speaker"] == "v5_cis_base"
        assert kwargs["trust_repo"] is True

    def test_load_moves_to_cuda_when_available(self, mock_torch_hub):
        """GPU-bound model is the prod configuration on the L4 host."""
        fake_model = MagicMock(name="silero_model")
        # Simulate .to(cuda) returning a possibly-different object — the
        # cached singleton should be the post-move instance.
        moved_model = MagicMock(name="silero_model_on_cuda")
        fake_model.to.return_value = moved_model
        mock_torch_hub.hub_load.return_value = (fake_model, {})
        mock_torch_hub.cuda_available.return_value = True

        result = load_silero()

        assert result is moved_model
        fake_model.to.assert_called_once()

    def test_load_stays_on_cpu_when_no_cuda(self, mock_torch_hub):
        fake_model = MagicMock(name="silero_model")
        mock_torch_hub.hub_load.return_value = (fake_model, {})
        mock_torch_hub.cuda_available.return_value = False

        result = load_silero()

        assert result is fake_model
        fake_model.to.assert_not_called()

    def test_get_silero_before_load_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            get_silero()

    def test_get_silero_error_message_mentions_load_silero(self):
        """Operators need to see the fix in the error text."""
        with pytest.raises(RuntimeError, match="call load_silero"):
            get_silero()

    def test_load_propagates_exceptions(self, mock_torch_hub):
        mock_torch_hub.hub_load.side_effect = RuntimeError("network down")

        with pytest.raises(RuntimeError, match="network down"):
            load_silero()


# ─── SileroTTSService.run_tts ────────────────────────────────────────


class TestRunTts:
    def test_happy_path_yields_audio_frame(self):
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        tensor = _make_audio_tensor(n_samples=12000)
        svc, fake_model = _build_service(apply_tts_return=tensor)

        frames = asyncio.run(_drain(svc.run_tts("привет мир", context_id="ctx-1")))

        # File-at-a-time: exactly one TTSAudioRawFrame is emitted for
        # the full utterance. (More would indicate the helper was
        # chunking, which we don't expect at 12 kB of audio.)
        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) == 1
        frame = audio_frames[0]
        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert len(frame.audio) > 0
        fake_model.apply_tts.assert_called_once()
        # apply_tts was invoked with the 24 kHz sample rate this service
        # negotiated with LiveKit — changing either side without the
        # other is a silent resample bug.
        apply_kwargs = fake_model.apply_tts.call_args.kwargs
        assert apply_kwargs["sample_rate"] == 24000
        assert apply_kwargs["text"] == "привет мир"
        assert apply_kwargs["put_accent"] is True
        assert apply_kwargs["put_yo"] is True

    def test_apply_tts_runs_off_the_event_loop(self):
        """asyncio.to_thread prevents the blocking synth from stalling I/O."""
        tensor = _make_audio_tensor()
        svc, fake_model = _build_service(apply_tts_return=tensor)

        # Record that apply_tts was called — the to_thread wrapper means
        # the call happens in a worker thread, not the main asyncio
        # loop, but the MagicMock is invoked regardless. Proving
        # non-blocking behaviour from a unit test without timing
        # instrumentation is infeasible; the call count gate is the
        # best we can do here. The mechanism is verified at the
        # integration level via the live deploy check in Unit 3's
        # verification steps.
        asyncio.run(_drain(svc.run_tts("hello", context_id="ctx")))
        fake_model.apply_tts.assert_called_once()

    def test_empty_text_drops_without_synth(self):
        svc, fake_model = _build_service(apply_tts_return=_make_audio_tensor())

        frames = asyncio.run(_drain(svc.run_tts("", context_id="ctx")))

        assert frames == []
        fake_model.apply_tts.assert_not_called()

    def test_whitespace_only_text_drops_without_synth(self):
        svc, fake_model = _build_service(apply_tts_return=_make_audio_tensor())

        frames = asyncio.run(_drain(svc.run_tts("   ", context_id="ctx")))

        assert frames == []
        fake_model.apply_tts.assert_not_called()

    def test_apply_tts_exception_yields_redacted_error_frame(self):
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc, fake_model = _build_service(apply_tts_return=_make_audio_tensor())
        fake_model.apply_tts.side_effect = RuntimeError("CUDA OOM tensor shape mismatch")

        frames = asyncio.run(_drain(svc.run_tts("привет", context_id="ctx")))

        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 1
        # The internal detail (CUDA OOM, tensor shape) must not leak to
        # the client-visible ErrorFrame — log server-side only.
        assert error_frames[0].error == "TTS synth failed"
        assert "CUDA" not in error_frames[0].error
        assert "OOM" not in error_frames[0].error

    def test_model_not_loaded_yields_error_frame(self):
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = SileroTTSService()  # no model injected
        svc.start_tts_usage_metrics = AsyncMock()
        svc.stop_ttfb_metrics = AsyncMock()
        svc._sample_rate = 24000

        frames = asyncio.run(_drain(svc.run_tts("привет", context_id="ctx")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        # Generic client-facing message — mirrors the gigaam_stt.py
        # precedent of logging internal cause server-side and emitting
        # a redacted message downstream.
        assert "TTS unavailable" in frames[0].error

    def test_custom_speaker_flows_into_tts_settings(self):
        """SileroSettings.speaker populates TTSSettings.voice at init time."""
        svc, _ = _build_service(
            apply_tts_return=_make_audio_tensor(),
            settings=SileroSettings(speaker="v5_custom_ru"),
        )

        # The framework TTSSettings store is populated so pipecat's
        # settings-tracking logs don't warn about NOT_GIVEN fields on
        # every pipeline start.
        assert svc._settings.model == "v5_custom_ru"
        # The adapter's own settings object also carries the speaker,
        # which is what load_silero / torch.hub.load would consume.
        assert svc._silero_settings.speaker == "v5_custom_ru"

    def test_default_settings_use_v5_cis_base(self):
        """Default SileroSettings match the spike's chosen model."""
        svc, _ = _build_service(apply_tts_return=_make_audio_tensor())

        assert svc._silero_settings.speaker == "v5_cis_base"
        assert svc._silero_settings.language == "ru"

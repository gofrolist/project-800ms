"""Tests for GigaAMSTTService + models.load_gigaam().

These tests mock the `gigaam` package entirely so they run on a CPU-only
CI runner (no CUDA) without needing to install the gigaam wheel or the
PyTorch extras. The real model integration happens on the GPU host
during Unit 3's WER smoke harness.

Convention: tests are sync and use `asyncio.run()` for async bodies,
matching the existing pattern in test_transcript_sink.py. The agent
test suite deliberately avoids pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# gigaam is not available in this CI environment — inject a stub module
# BEFORE any test imports gigaam_stt or models so the lazy `import gigaam`
# inside those modules resolves to our mock. This mirrors the pattern used
# elsewhere in the agent tests for heavy deps.
_gigaam_stub = MagicMock()
sys.modules.setdefault("gigaam", _gigaam_stub)

from gigaam_stt import (  # noqa: E402 — must import after stub injection
    GigaAMSettings,
    GigaAMSTTService,
)
from models import _reset_models_for_tests, get_gigaam, load_gigaam  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    _reset_models_for_tests()
    _gigaam_stub.reset_mock()
    _gigaam_stub.load_model.side_effect = None
    yield
    _reset_models_for_tests()


def _make_audio_bytes(duration_s: float, sample_rate: int = 16_000) -> bytes:
    """Synthesize `duration_s` seconds of silence as int16 PCM bytes."""
    n_samples = int(duration_s * sample_rate)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


async def _drain(async_gen):
    """Collect frames from an AsyncGenerator into a list."""
    return [frame async for frame in async_gen]


def _build_service(*, transcribe_return, settings=None):
    """Factory for a service backed by a mock gigaam model.

    Skips the full Pipecat start-processing ceremony (it awaits services
    the test runner hasn't initialized) by overriding the few hooks our
    code path actually calls.
    """
    fake_model = MagicMock(name="gigaam_model")
    fake_model.transcribe = MagicMock(return_value=transcribe_return)
    svc = GigaAMSTTService(model=fake_model, settings=settings)
    svc._loaded_model = fake_model
    svc.start_processing_metrics = AsyncMock()
    svc.stop_processing_metrics = AsyncMock()
    svc._handle_transcription = AsyncMock()
    svc._user_id = "test-user"
    return svc, fake_model


# ─── models.load_gigaam / get_gigaam ─────────────────────────────────


class TestLoadGigaam:
    def test_loads_once_and_caches(self):
        fake_model = MagicMock(name="gigaam_model")
        _gigaam_stub.load_model.return_value = fake_model

        first = load_gigaam("v3_ctc")
        second = load_gigaam("v3_ctc")

        assert first is fake_model
        assert second is fake_model
        # Cached singleton — gigaam.load_model called exactly once.
        assert _gigaam_stub.load_model.call_count == 1

    def test_get_gigaam_before_load_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            get_gigaam()

    def test_load_propagates_exceptions(self):
        _gigaam_stub.load_model.side_effect = RuntimeError("HF download failed")

        with pytest.raises(RuntimeError, match="HF download failed"):
            load_gigaam()


# ─── GigaAMSTTService.run_stt ─────────────────────────────────────────


class TestRunStt:
    def test_happy_path_yields_transcription_frame(self):
        from pipecat.frames.frames import TranscriptionFrame

        svc, fake_model = _build_service(transcribe_return="привет как дела")
        audio = _make_audio_bytes(1.0)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == "привет как дела"
        fake_model.transcribe.assert_called_once()

    def test_duration_below_threshold_drops(self):
        """Default MIN_DURATION=0.30s; feed 0.1s → must drop before decode."""
        svc, fake_model = _build_service(transcribe_return="x")
        audio = _make_audio_bytes(0.1)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert frames == []
        fake_model.transcribe.assert_not_called()  # short-circuits before decode

    def test_token_count_below_threshold_drops(self):
        """Default MIN_TOKEN=2; transcribe returns 1 word → drop."""
        svc, _ = _build_service(transcribe_return="привет")
        audio = _make_audio_bytes(1.0)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert frames == []

    def test_empty_transcribe_result_drops(self):
        svc, _ = _build_service(transcribe_return="")
        audio = _make_audio_bytes(1.0)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert frames == []

    def test_transcribe_exception_yields_error_frame(self):
        from pipecat.frames.frames import ErrorFrame

        svc, fake_model = _build_service(transcribe_return="unused")
        fake_model.transcribe.side_effect = RuntimeError("CUDA OOM")
        audio = _make_audio_bytes(1.0)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert "CUDA OOM" in frames[0].error

    def test_model_not_loaded_yields_error_frame(self):
        from pipecat.frames.frames import ErrorFrame

        svc = GigaAMSTTService()  # no model injected
        svc.start_processing_metrics = AsyncMock()
        svc.stop_processing_metrics = AsyncMock()
        # simulate "_load was never called" → _loaded_model stays None

        frames = asyncio.run(_drain(svc.run_stt(_make_audio_bytes(1.0))))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert "not available" in frames[0].error

    def test_normalizes_result_with_text_attribute(self):
        """Some gigaam return shapes expose `.text`; verify we unwrap."""
        from pipecat.frames.frames import TranscriptionFrame

        result_obj = MagicMock()
        result_obj.text = "привет мир"
        svc, _ = _build_service(transcribe_return=result_obj)
        audio = _make_audio_bytes(1.0)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == "привет мир"

    def test_custom_settings_override_defaults(self):
        """Filter thresholds tunable per-instance for Unit 3's parity pass."""
        from pipecat.frames.frames import TranscriptionFrame

        svc, _ = _build_service(
            transcribe_return="привет",
            settings=GigaAMSettings(min_duration_seconds=0.05, min_token_count=1),
        )
        audio = _make_audio_bytes(0.1)

        frames = asyncio.run(_drain(svc.run_stt(audio)))

        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)

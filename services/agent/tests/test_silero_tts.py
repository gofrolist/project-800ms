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

import pytest
import torch

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


def _make_audio_tensor(n_samples: int = 24000) -> torch.Tensor:
    """Build a real torch.Tensor mirroring Silero's apply_tts return type.

    Silero returns a float32 torch.Tensor in [-1, 1]. Using a real
    tensor — not a MagicMock — ensures the ``.detach().cpu().numpy()``
    chain the adapter invokes is actually exercised in tests; a
    MagicMock would fake any attribute access and mask regressions
    like a bare ``.numpy()`` call against a CUDA tensor.
    """
    return torch.linspace(-0.5, 0.5, n_samples, dtype=torch.float32)


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
        # two load_silero() invocations with the same model name.
        assert mock_torch_hub.hub_load.call_count == 1

    def test_second_load_with_different_name_raises(self, mock_torch_hub):
        """Switching speakers at runtime must not silently serve the
        already-loaded model.

        Upstream behavior returned the cached model regardless of
        ``model_name``, which masked config drift. We raise explicitly
        so the operator sees the mismatch on agent boot.
        """
        fake_model = MagicMock(name="silero_model")
        mock_torch_hub.hub_load.return_value = (fake_model, {})

        load_silero("v5_cis_base")

        with pytest.raises(RuntimeError) as exc_info:
            load_silero("v5_ru")

        message = str(exc_info.value)
        # Both the cached name and the requested name appear in the
        # error so the operator can see what's loaded and what they
        # asked for.
        assert "'v5_cis_base'" in message
        assert "'v5_ru'" in message
        # The second call must NOT hit torch.hub.load — otherwise we
        # would download and discard a model before raising.
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
        """GPU-bound model is the prod configuration on the L4 host.

        Regression guard for the silent-model-None bug
        (fix commit 49a4d4f): Silero v5 ships as a torch.package scripted
        module whose ``.to(device)`` returns ``None`` instead of ``self``.
        An earlier version of models.load_silero() did
        ``model = model.to(device)``, which silently replaced the model
        with ``None`` in prod — every Silero synth attempt then emitted
        ``ErrorFrame("TTS unavailable")``. The test that lived here
        *encoded* the bug by having ``fake_model.to.return_value`` stand
        in as a ``moved_model`` the caller could reassign — that made the
        buggy code green.

        The correct contract is: ``.to(device)`` is called for its
        side-effects only, and the cached singleton is the same
        ``fake_model`` instance torch.hub.load returned. This test now
        mirrors that: ``.to(device)`` is asserted-called with the cuda
        device, but its return value is ignored.
        """
        import torch  # noqa: PLC0415

        fake_model = MagicMock(name="silero_model")
        # Explicitly return ``None`` from ``.to(device)`` — matches real
        # Silero v5 behaviour. If the implementation reassigns the model
        # from the ``.to()`` return value, the cached singleton would be
        # ``None`` and this test would fail on the `is fake_model` check.
        fake_model.to.return_value = None
        mock_torch_hub.hub_load.return_value = (fake_model, {})
        mock_torch_hub.cuda_available.return_value = True

        result = load_silero()

        assert result is fake_model
        # ``.to()`` was called for its side effect with a cuda device.
        fake_model.to.assert_called_once()
        ((device_arg,), _kwargs) = fake_model.to.call_args
        assert isinstance(device_arg, torch.device)
        assert device_arg.type == "cuda"

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
        # Regression guard: adapter MUST pass speaker — v5_cis_base's
        # apply_tts silently falls back to its default speaker
        # (ru_zhadyra) when speaker= is omitted, so a missing kwarg
        # would look like "voice picker broken" from the user's POV.
        # Default SileroSettings.speaker = "ru_zhadyra" flows into
        # TTSSettings.voice which run_tts reads.
        assert apply_kwargs["speaker"] == "ru_zhadyra"

    def test_real_torch_tensor_survives_detach_cpu_numpy_chain(self):
        """Using a genuine torch.Tensor (not a MagicMock) exercises the
        ``.detach().cpu().numpy()`` conversion path end-to-end.

        A ``MagicMock`` fakes any attribute access, which would silently
        succeed even if the adapter called a non-existent chain. Using
        a real CPU tensor asserts the actual method chain works and
        produces non-empty PCM output — and would fail loudly if
        someone regressed to bare ``.numpy()`` against a CUDA-resident
        tensor, because the same call path is what prod would hit.
        """
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        real_tensor = torch.zeros(12000, dtype=torch.float32)
        # Fill with a small non-zero ramp so the int16 conversion
        # produces non-trivial PCM bytes the framing helper won't drop.
        real_tensor += torch.linspace(-0.5, 0.5, 12000)

        svc, fake_model = _build_service(apply_tts_return=real_tensor)

        frames = asyncio.run(_drain(svc.run_tts("hello", context_id="ctx")))

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) == 1
        assert len(audio_frames[0].audio) > 0
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

    def test_stop_ttfb_metrics_fires_exactly_once_on_happy_path(self):
        """A completed synth calls stop_ttfb_metrics exactly once.

        Previously the method was called inside the async-for loop AND
        in ``finally``, so a single happy-path synth incremented the
        stop counter twice. The sentinel-flag pattern must gate the
        ``finally`` branch so it never duplicates work.
        """
        svc, _ = _build_service(apply_tts_return=_make_audio_tensor())

        asyncio.run(_drain(svc.run_tts("привет мир", context_id="ctx")))

        assert svc.stop_ttfb_metrics.call_count == 1

    def test_concurrent_apply_tts_calls_are_serialized(self):
        """Two concurrent ``run_tts`` calls against the shared Silero
        singleton must NOT execute ``apply_tts`` in parallel.

        PyTorch ``nn.Module`` isn't thread-safe; interleaved calls on
        the same module via the threadpool executor can corrupt hidden
        state. The module-level ``_silero_apply_lock`` serializes
        access. We prove serialization by tracking in-flight counts
        inside the mock — if the lock is removed, the max in-flight
        count climbs to 2 and the assertion fails.
        """
        import threading  # noqa: PLC0415

        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()
        start_event = threading.Event()

        def fake_apply_tts(**_kwargs):
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            # Hold the "GPU" long enough for the concurrent call to
            # reach the same critical section if the lock weren't
            # there. Short enough that tests don't run slow.
            start_event.wait(timeout=0.1)
            with lock:
                in_flight -= 1
            return torch.linspace(-0.5, 0.5, 12000, dtype=torch.float32)

        svc_a, fake_model_a = _build_service(apply_tts_return=None)
        fake_model_a.apply_tts = fake_apply_tts
        svc_b, fake_model_b = _build_service(apply_tts_return=None)
        fake_model_b.apply_tts = fake_apply_tts

        async def _race():
            # Release the waiting thread after both run_tts calls have
            # been kicked off — otherwise the first call finishes
            # before the second starts and we don't test contention.
            async def _release_after_both_started():
                await asyncio.sleep(0.02)
                start_event.set()

            asyncio.get_event_loop().create_task(_release_after_both_started())
            await asyncio.gather(
                _drain(svc_a.run_tts("hello a", context_id="ctx-a")),
                _drain(svc_b.run_tts("hello b", context_id="ctx-b")),
            )

        asyncio.run(_race())

        # If the lock were absent, both apply_tts calls would enter
        # fake_apply_tts concurrently and max_in_flight would reach 2.
        assert max_in_flight == 1, (
            f"expected serialized apply_tts calls, got max_in_flight={max_in_flight}"
        )

    def test_stop_ttfb_metrics_fires_exactly_once_on_error_path(self):
        """When apply_tts raises before any frame yields, stop_ttfb_metrics
        must still fire exactly once (from the ``finally`` branch)."""
        svc, fake_model = _build_service(apply_tts_return=_make_audio_tensor())
        fake_model.apply_tts.side_effect = RuntimeError("boom")

        asyncio.run(_drain(svc.run_tts("привет", context_id="ctx")))

        assert svc.stop_ttfb_metrics.call_count == 1

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
        """SileroSettings.speaker populates TTSSettings at init time.

        The SileroSettings instance itself is consumed eagerly in
        ``__init__`` (no stored reference), so the assertion moves to
        the framework's TTSSettings store which IS the single source
        of truth after construction.
        """
        svc, _ = _build_service(
            apply_tts_return=_make_audio_tensor(),
            settings=SileroSettings(speaker="v5_custom_ru"),
        )

        # The framework TTSSettings store is populated so pipecat's
        # settings-tracking logs don't warn about NOT_GIVEN fields on
        # every pipeline start. TTSSettings.voice + .model both carry
        # the speaker id (SileroSettings echoes it into both).
        assert svc._settings.model == "v5_custom_ru"
        assert svc._settings.voice == "v5_custom_ru"

    def test_default_settings_use_ru_zhadyra_speaker(self):
        """Default SileroSettings.speaker matches v5_cis_base's own
        apply_tts default (ru_zhadyra). Keeps behavior unchanged for
        deploys that don't set a speaker explicitly."""
        svc, _ = _build_service(apply_tts_return=_make_audio_tensor())

        # SileroSettings.speaker is what apply_tts(speaker=...) reads
        # at synth time — NOT the torch.hub.load model name.
        assert svc._settings.model == "ru_zhadyra"
        assert svc._settings.voice == "ru_zhadyra"
        assert svc._settings.language == "ru"

    def test_speaker_flows_to_apply_tts(self):
        """Setting SileroSettings.speaker must reach apply_tts(speaker=...)
        — otherwise the model silently falls back to its built-in default
        (ru_zhadyra) and the voice picker is a no-op from the user's
        POV."""
        svc, fake_model = _build_service(
            apply_tts_return=_make_audio_tensor(),
            settings=SileroSettings(speaker="ru_dmitriy"),
        )

        asyncio.run(_drain(svc.run_tts("текст", "ctx")))

        fake_model.apply_tts.assert_called_once()
        assert fake_model.apply_tts.call_args.kwargs["speaker"] == "ru_dmitriy"

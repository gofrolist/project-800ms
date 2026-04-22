"""Tests for XTTSTTSService + models.load_xtts / get_xtts.

Same mocking shape as test_silero_tts.py:

* gigaam is stubbed at module scope so shared imports through ``models``
  don't fail on the CPU-only CI runner.
* torch IS installed in the agent venv (gigaam[torch] pulls it), but
  coqui-tts is NOT — we stub ``TTS.api.TTS`` at the lazy-import site
  inside ``models.load_xtts`` to avoid the network download that the
  real constructor performs on first use.
* Tests are sync and use ``asyncio.run()`` for async bodies. Agent test
  suite deliberately avoids pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# gigaam lives alongside silero + xtts in models.py; stub before importing.
sys.modules.setdefault("gigaam", MagicMock())

from models import (  # noqa: E402 — must import after stub injection
    _reset_models_for_tests,
    get_xtts,
    load_xtts,
)
from xtts_tts import (  # noqa: E402
    _DEFAULT_XTTS_LANGUAGE,
    XTTSSettings,
    XTTSTTSService,
    _resolve_voice_profile,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_models_for_tests()
    yield
    _reset_models_for_tests()


# ─── _resolve_voice_profile ──────────────────────────────────────────


class TestResolveVoiceProfile:
    """Every invalid input path must raise ValueError at construction
    time — otherwise a bad voice config would surface as silence (or an
    ErrorFrame) only after the user has already spoken."""

    def _build_profile(
        self,
        base: Path,
        *,
        profile_id: str = "demo-ru",
        meta: dict | None = None,
        write_ref_audio: bool = True,
        ref_filename: str = "ref.wav",
    ) -> Path:
        """Materialize a voice-library profile on disk for the tests."""
        profile_dir = base / "profiles" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        if write_ref_audio:
            (profile_dir / ref_filename).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        if meta is not None:
            (profile_dir / "meta.json").write_text(
                json.dumps(meta),
                encoding="utf-8",
            )
        return profile_dir

    def test_happy_path_russian_profile(self, tmp_path):
        self._build_profile(
            tmp_path,
            meta={
                "profile_id": "demo-ru",
                "ref_audio_filename": "ref.wav",
                "language": "Russian",
            },
        )
        resolved = _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

        assert resolved.profile_id == "demo-ru"
        assert resolved.ref_audio_path == tmp_path / "profiles" / "demo-ru" / "ref.wav"
        # "Russian" in meta.json → "ru" XTTS code.
        assert resolved.language == "ru"

    def test_language_already_in_xtts_code_form(self, tmp_path):
        self._build_profile(
            tmp_path,
            meta={"ref_audio_filename": "ref.wav", "language": "ru"},
        )
        resolved = _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

        assert resolved.language == "ru"

    def test_missing_language_field_falls_back_to_default(self, tmp_path):
        self._build_profile(
            tmp_path,
            meta={"ref_audio_filename": "ref.wav"},
        )
        resolved = _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

        assert resolved.language == _DEFAULT_XTTS_LANGUAGE

    def test_unknown_language_string_falls_back_to_default(self, tmp_path):
        """An unrecognized language value must not crash the adapter — it
        degrades to the default (Russian) and the operator sees the
        wrong-language output in QA. Failing closed here would break deploy.
        """
        self._build_profile(
            tmp_path,
            meta={"ref_audio_filename": "ref.wav", "language": "Klingon"},
        )
        resolved = _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

        assert resolved.language == _DEFAULT_XTTS_LANGUAGE

    def test_custom_ref_filename_respected(self, tmp_path):
        """Operators can set ref_audio_filename to anything — the adapter
        must not hard-code ``ref.wav``."""
        self._build_profile(
            tmp_path,
            meta={"ref_audio_filename": "voice_sample.wav", "language": "ru"},
            ref_filename="voice_sample.wav",
        )
        resolved = _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

        assert resolved.ref_audio_path.name == "voice_sample.wav"

    def test_voice_without_clone_prefix_raises(self, tmp_path):
        """XTTS is voice-cloning only — a bare speaker id (e.g. a Silero
        name) must be rejected at init, not at synth."""
        with pytest.raises(ValueError, match="clone:<profile>"):
            _resolve_voice_profile("ru_dmitriy", voice_library_dir=tmp_path)

    def test_empty_profile_id_raises(self, tmp_path):
        with pytest.raises(ValueError, match="profile id is empty"):
            _resolve_voice_profile("clone:", voice_library_dir=tmp_path)

    def test_missing_meta_json_raises(self, tmp_path):
        # Create the profile dir but no meta.json.
        (tmp_path / "profiles" / "demo-ru").mkdir(parents=True)
        with pytest.raises(ValueError, match="meta.json not found"):
            _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

    def test_missing_profile_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="meta.json not found"):
            _resolve_voice_profile("clone:does-not-exist", voice_library_dir=tmp_path)

    def test_malformed_json_raises(self, tmp_path):
        profile_dir = tmp_path / "profiles" / "demo-ru"
        profile_dir.mkdir(parents=True)
        (profile_dir / "meta.json").write_text("not { valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to read"):
            _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

    def test_non_object_meta_raises(self, tmp_path):
        """meta.json must be a JSON object — an array or scalar breaks
        every downstream .get() call, so fail loudly."""
        profile_dir = tmp_path / "profiles" / "demo-ru"
        profile_dir.mkdir(parents=True)
        (profile_dir / "meta.json").write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="not a JSON object"):
            _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)

    def test_missing_ref_audio_raises(self, tmp_path):
        """meta.json points to a ref file that doesn't exist → raise,
        don't silently try to synth with a missing speaker_wav."""
        self._build_profile(
            tmp_path,
            meta={"ref_audio_filename": "ref.wav", "language": "ru"},
            write_ref_audio=False,
        )

        with pytest.raises(ValueError, match="ref audio not found"):
            _resolve_voice_profile("clone:demo-ru", voice_library_dir=tmp_path)


# ─── models.load_xtts / get_xtts ─────────────────────────────────────


class TestLoadXtts:
    """Cover the same cache/device/error shape as load_silero."""

    def _patch_tts_import(self, monkeypatch, tts_cls_mock: MagicMock) -> None:
        """Stub ``TTS.api.TTS`` at the lazy-import site inside load_xtts.

        The real ``TTS.api`` constructor tries to reach HuggingFace for
        the XTTS v2 checkpoint, which we can't do in tests. Registering
        fake modules under ``TTS`` + ``TTS.api`` in ``sys.modules``
        forces ``from TTS.api import TTS`` inside ``load_xtts`` to
        resolve to our mock class.
        """
        fake_api = MagicMock(name="TTS.api")
        fake_api.TTS = tts_cls_mock
        fake_pkg = MagicMock(name="TTS", api=fake_api)
        monkeypatch.setitem(sys.modules, "TTS", fake_pkg)
        monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    def test_loads_once_and_caches(self, monkeypatch):
        import torch  # noqa: PLC0415

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        first = load_xtts()
        second = load_xtts()

        assert first is fake_tts
        assert second is fake_tts
        # Single construction across two calls with the same name.
        assert tts_cls.call_count == 1

    def test_second_load_with_different_name_raises(self, monkeypatch):
        import torch  # noqa: PLC0415

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        load_xtts("tts_models/multilingual/multi-dataset/xtts_v2")
        with pytest.raises(RuntimeError) as exc_info:
            load_xtts("tts_models/different/xtts_custom")

        message = str(exc_info.value)
        assert "'tts_models/multilingual/multi-dataset/xtts_v2'" in message
        assert "'tts_models/different/xtts_custom'" in message
        # Second call must NOT construct a second TTS instance.
        assert tts_cls.call_count == 1

    def test_load_moves_to_cuda_when_available(self, monkeypatch):
        import torch  # noqa: PLC0415

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        result = load_xtts()

        assert result is fake_tts
        # ``.to(device)`` called for its side-effect with a cuda device.
        fake_tts.to.assert_called_once()
        ((device_arg,), _kwargs) = fake_tts.to.call_args
        assert isinstance(device_arg, torch.device)
        assert device_arg.type == "cuda"

    def test_load_stays_on_cpu_when_no_cuda(self, monkeypatch):
        import torch  # noqa: PLC0415

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = load_xtts()

        assert result is fake_tts
        fake_tts.to.assert_not_called()

    def test_get_xtts_before_load_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            get_xtts()

    def test_get_xtts_error_message_mentions_load_xtts(self):
        """Operators need to see the fix in the error text."""
        with pytest.raises(RuntimeError, match="call load_xtts"):
            get_xtts()


# ─── XTTSTTSService.run_tts ──────────────────────────────────────────


def _drain(async_gen):
    """Collect frames from an AsyncGenerator into a list."""

    async def _go():
        return [frame async for frame in async_gen]

    return asyncio.run(_go())


def _make_profile(base: Path, *, profile_id: str = "demo-ru") -> Path:
    """Build a minimal voice-library profile on disk."""
    profile_dir = base / "profiles" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "ref.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (profile_dir / "meta.json").write_text(
        json.dumps({"ref_audio_filename": "ref.wav", "language": "ru"}),
        encoding="utf-8",
    )
    return profile_dir


def _build_service(
    *,
    voice_library_dir: Path,
    tts_return: object,
    voice: str = "clone:demo-ru",
):
    """Factory for an XTTSTTSService backed by a mocked coqui ``TTS``."""
    fake_model = MagicMock(name="xtts_model")
    fake_model.tts = MagicMock(return_value=tts_return)
    svc = XTTSTTSService(
        xtts_model=fake_model,
        settings=XTTSSettings(voice=voice, voice_library_dir=voice_library_dir),
    )
    svc._loaded_model = fake_model
    svc.start_tts_usage_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc._sample_rate = svc._init_sample_rate or 24000
    return svc, fake_model


class TestRunTts:
    def test_happy_path_yields_audio_frame(self, tmp_path):
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        _make_profile(tmp_path)
        # Coqui TTS.api.tts() typically returns list[float]; give a
        # realistic input that exercises np.asarray + int16 conversion.
        audio = np.linspace(-0.5, 0.5, 12000, dtype=np.float32).tolist()
        svc, fake_model = _build_service(voice_library_dir=tmp_path, tts_return=audio)

        frames = _drain(svc.run_tts("привет мир", context_id="ctx"))

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) == 1
        frame = audio_frames[0]
        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert len(frame.audio) > 0
        fake_model.tts.assert_called_once()
        kwargs = fake_model.tts.call_args.kwargs
        assert kwargs["text"] == "привет мир"
        # Language was resolved from the profile's meta.json.
        assert kwargs["language"] == "ru"
        # speaker_wav is a string (coqui expects a path, not a Path
        # object — the adapter must str() it).
        assert isinstance(kwargs["speaker_wav"], str)
        assert kwargs["speaker_wav"].endswith("ref.wav")

    def test_ndarray_return_type_also_works(self, tmp_path):
        """coqui-tts sometimes returns np.ndarray instead of list[float].
        np.asarray handles both; this pins that we don't rely on list-ness.
        """
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        _make_profile(tmp_path)
        audio = np.linspace(-0.3, 0.3, 8000, dtype=np.float32)
        svc, _ = _build_service(voice_library_dir=tmp_path, tts_return=audio)

        frames = _drain(svc.run_tts("hello", context_id="ctx"))

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) == 1
        assert len(audio_frames[0].audio) > 0

    def test_empty_text_drops_without_synth(self, tmp_path):
        _make_profile(tmp_path)
        svc, fake_model = _build_service(
            voice_library_dir=tmp_path,
            tts_return=[0.0] * 12000,
        )

        frames = _drain(svc.run_tts("", context_id="ctx"))

        assert frames == []
        fake_model.tts.assert_not_called()

    def test_whitespace_only_text_drops_without_synth(self, tmp_path):
        _make_profile(tmp_path)
        svc, fake_model = _build_service(
            voice_library_dir=tmp_path,
            tts_return=[0.0] * 12000,
        )

        frames = _drain(svc.run_tts("   ", context_id="ctx"))

        assert frames == []
        fake_model.tts.assert_not_called()

    def test_tts_exception_yields_redacted_error_frame(self, tmp_path):
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        _make_profile(tmp_path)
        svc, fake_model = _build_service(
            voice_library_dir=tmp_path,
            tts_return=[0.0] * 12000,
        )
        fake_model.tts.side_effect = RuntimeError("CUDA OOM on conditioning latents")

        frames = _drain(svc.run_tts("привет", context_id="ctx"))

        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 1
        # Internal detail must not leak to client.
        assert error_frames[0].error == "TTS synth failed"
        assert "CUDA" not in error_frames[0].error
        assert "OOM" not in error_frames[0].error

    def test_stop_ttfb_metrics_fires_exactly_once_on_happy_path(self, tmp_path):
        """A completed synth calls stop_ttfb_metrics exactly once.

        Same sentinel-flag pattern Silero uses — verify the finally
        branch doesn't double-count after the async-for already stopped
        the timer.
        """
        _make_profile(tmp_path)
        audio = np.linspace(-0.5, 0.5, 12000, dtype=np.float32).tolist()
        svc, _ = _build_service(voice_library_dir=tmp_path, tts_return=audio)

        _drain(svc.run_tts("привет мир", context_id="ctx"))

        assert svc.stop_ttfb_metrics.call_count == 1

    def test_stop_ttfb_metrics_fires_exactly_once_on_error_path(self, tmp_path):
        _make_profile(tmp_path)
        svc, fake_model = _build_service(
            voice_library_dir=tmp_path,
            tts_return=[0.0] * 12000,
        )
        fake_model.tts.side_effect = RuntimeError("boom")

        _drain(svc.run_tts("привет", context_id="ctx"))

        assert svc.stop_ttfb_metrics.call_count == 1

    def test_model_not_loaded_yields_error_frame(self, tmp_path):
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        _make_profile(tmp_path)
        # Construct normally then null-out the loaded model — mirrors the
        # Silero test, which exercises the defensive check against a
        # production deploy that somehow booted without the preload.
        svc = XTTSTTSService(
            xtts_model=MagicMock(name="xtts_placeholder"),
            settings=XTTSSettings(voice="clone:demo-ru", voice_library_dir=tmp_path),
        )
        svc._loaded_model = None
        svc.start_tts_usage_metrics = AsyncMock()
        svc.stop_ttfb_metrics = AsyncMock()
        svc._sample_rate = 24000

        frames = _drain(svc.run_tts("привет", context_id="ctx"))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert "TTS unavailable" in frames[0].error

    def test_concurrent_tts_calls_are_serialized(self, tmp_path):
        """Two concurrent ``run_tts`` calls against the shared coqui TTS
        singleton must NOT execute ``tts()`` in parallel — same hidden-
        state corruption risk as Silero.
        """
        import threading  # noqa: PLC0415

        _make_profile(tmp_path)

        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()
        start_event = threading.Event()

        def fake_tts(**_kwargs):
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            start_event.wait(timeout=0.1)
            with lock:
                in_flight -= 1
            return [0.1] * 12000

        svc_a, fake_model_a = _build_service(
            voice_library_dir=tmp_path,
            tts_return=None,
        )
        fake_model_a.tts = fake_tts
        svc_b, fake_model_b = _build_service(
            voice_library_dir=tmp_path,
            tts_return=None,
        )
        fake_model_b.tts = fake_tts

        async def _race():
            async def _release_after_both_started():
                await asyncio.sleep(0.02)
                start_event.set()

            asyncio.get_event_loop().create_task(_release_after_both_started())

            async def _collect(gen):
                return [frame async for frame in gen]

            await asyncio.gather(
                _collect(svc_a.run_tts("a", context_id="ctx-a")),
                _collect(svc_b.run_tts("b", context_id="ctx-b")),
            )

        asyncio.run(_race())

        assert max_in_flight == 1, (
            f"expected serialized tts() calls, got max_in_flight={max_in_flight}"
        )


class TestInitValidation:
    """Profile resolution at __init__ time — a bad voice config must
    fail during service construction (which pipeline.build_task runs on
    dispatch) rather than at first synth, so the user gets a clean
    /dispatch failure instead of the agent playing silence."""

    def test_bad_voice_raises_at_init(self, tmp_path):
        with pytest.raises(ValueError, match="clone:<profile>"):
            XTTSTTSService(
                xtts_model=MagicMock(),
                settings=XTTSSettings(voice="bare_voice", voice_library_dir=tmp_path),
            )

    def test_missing_profile_raises_at_init(self, tmp_path):
        with pytest.raises(ValueError):
            XTTSTTSService(
                xtts_model=MagicMock(),
                settings=XTTSSettings(voice="clone:does-not-exist", voice_library_dir=tmp_path),
            )

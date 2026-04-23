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
import time
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
    _xtts_synth_idle,
    XTTSSettings,
    XTTSTTSService,
    _resolve_voice_profile,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_models_for_tests()
    # Reset the synth-thread idle gate to "idle" — a prior test that
    # simulated cancellation without running the thread's finally clause
    # could leave this clear, which would hang the next synth forever
    # on `await asyncio.to_thread(_xtts_synth_idle.wait)`.
    _xtts_synth_idle.set()
    yield
    _reset_models_for_tests()
    _xtts_synth_idle.set()


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

    def test_profile_id_traversal_rejected(self, tmp_path):
        """A profile_id containing ``..`` must not escape the voice library
        root. Python's ``Path /`` operator does NOT sanitize ``..``; without
        the resolve() + relative_to() guard, ``clone:../../etc/passwd``
        would resolve outside ``voice_library_dir`` and stat arbitrary
        filesystem paths.
        """
        # A traversal that clearly escapes tmp_path. relative_to raises
        # as soon as the resolved path isn't under voice_library_dir,
        # independent of whether the target actually exists on disk —
        # the guard fires before any filesystem stat.
        with pytest.raises(ValueError, match="escapes voice library root"):
            _resolve_voice_profile(
                "clone:../../../etc/passwd",
                voice_library_dir=tmp_path,
            )

    def test_ref_audio_filename_traversal_rejected(self, tmp_path):
        """Second-stage guard: an operator-controlled meta.json with
        ``ref_audio_filename`` containing ``..`` must not let the ref audio
        path escape the profile directory.
        """
        # Set up a profile dir AND a sibling file that the traversal would
        # reach. On the happy path the resolve chain would find the sibling
        # file (is_file=True) and hand it to ``tts()`` as speaker_wav, which
        # is information disclosure (the file content flows into XTTS's
        # audio decoder). The guard must reject before is_file() is reached.
        self._build_profile(
            tmp_path,
            meta={
                "ref_audio_filename": "../external-leak.wav",
                "language": "ru",
            },
        )
        (tmp_path / "profiles" / "external-leak.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with pytest.raises(ValueError, match="escapes profile directory"):
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
        # Pin the XTTS v2 model-name argument so a regression that
        # swaps the hard-coded string or changes the positional/keyword
        # contract is caught here. The default argument on load_xtts()
        # must match what the coqui-tts TTS() constructor expects.
        tts_cls.assert_called_once_with("tts_models/multilingual/multi-dataset/xtts_v2")

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

    def test_sentinel_written_after_successful_load(self, monkeypatch, tmp_path):
        """load_xtts() must write .xtts-download-complete in the cache dir
        after TTS() returns cleanly, so subsequent boots can tell this
        cache from a partial one.

        The fake TTS() constructor creates the cache dir as a side
        effect, matching real coqui-tts behavior (it downloads into the
        cache dir during __init__). The sentinel is then written by
        load_xtts after the constructor returns cleanly.
        """
        import torch  # noqa: PLC0415

        monkeypatch.setenv("TTS_HOME", str(tmp_path))
        cache_dir = tmp_path / "tts_models--multilingual--multi-dataset--xtts_v2"

        def _tts_side_effect(*_args, **_kwargs):
            # Simulate coqui-tts creating the cache dir during its
            # download + materialization phase.
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "model.pth").write_bytes(b"fake weights")
            return MagicMock(name="xtts_model")

        tts_cls = MagicMock(name="TTS", side_effect=_tts_side_effect)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        load_xtts()

        sentinel = cache_dir / ".xtts-download-complete"
        assert sentinel.exists(), (
            f"expected sentinel {sentinel} to exist after load_xtts(); "
            f"dir contents: {list(cache_dir.iterdir()) if cache_dir.exists() else 'dir-missing'}"
        )

    def test_partial_download_is_cleaned_on_next_boot(self, monkeypatch, tmp_path):
        """Regression guard: a cache dir present WITHOUT the sentinel
        is treated as a partial download and cleared before TTS() runs.

        Without this self-heal, coqui-tts would trust the stale dir and
        torch.load() inside TTS() would raise FileNotFoundError on the
        missing weight file — permanent crashloop until manual cache wipe.
        """
        import torch  # noqa: PLC0415

        monkeypatch.setenv("TTS_HOME", str(tmp_path))
        cache_dir = tmp_path / "tts_models--multilingual--multi-dataset--xtts_v2"
        cache_dir.mkdir()
        stale_weight = cache_dir / "model.pth"
        stale_weight.write_bytes(b"partial download junk")
        # No sentinel file — this is the corruption state.

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        load_xtts()

        # The stale weight was removed (via rmtree of cache_dir), and
        # nothing pre-existed to re-create it — after rmtree the dir
        # is gone entirely since TTS() is mocked and doesn't re-download.
        assert not stale_weight.exists()

    def test_sentinel_present_skips_cleanup(self, monkeypatch, tmp_path):
        """If a previous successful load wrote the sentinel, subsequent
        loads must NOT touch the cache dir — the cache is valid."""
        import torch  # noqa: PLC0415

        monkeypatch.setenv("TTS_HOME", str(tmp_path))
        cache_dir = tmp_path / "tts_models--multilingual--multi-dataset--xtts_v2"
        cache_dir.mkdir()
        weight = cache_dir / "model.pth"
        weight.write_bytes(b"valid cached weights")
        (cache_dir / ".xtts-download-complete").touch()

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        load_xtts()

        # Weight was preserved (not cleared).
        assert weight.exists()
        assert weight.read_bytes() == b"valid cached weights"

    def test_missing_cache_dir_does_not_raise(self, monkeypatch, tmp_path):
        """Fresh deploy: TTS_HOME is set but the model cache dir doesn't
        exist yet. Self-heal and sentinel-write must both no-op cleanly."""
        import torch  # noqa: PLC0415

        monkeypatch.setenv("TTS_HOME", str(tmp_path))
        # Don't create the cache dir — simulates first-ever boot before
        # coqui-tts has downloaded anything. Our mocked TTS() doesn't
        # create it either.

        fake_tts = MagicMock(name="xtts_model")
        tts_cls = MagicMock(name="TTS", return_value=fake_tts)
        self._patch_tts_import(monkeypatch, tts_cls)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Must not raise despite the cache dir being absent.
        load_xtts()


# ─── XTTSTTSService.run_tts ──────────────────────────────────────────


def _drain(async_gen):
    """Collect frames from an AsyncGenerator into a list."""

    async def _go():
        return [frame async for frame in async_gen]

    return asyncio.run(_go())


async def _drain_async(async_gen):
    """Async form of ``_drain`` for use inside an existing event loop.

    The sync ``_drain`` wraps in ``asyncio.run``, which fails if called
    from inside a running loop. Tests that already create tasks with
    ``asyncio.create_task`` (e.g. the cancellation race test) need
    the plain async collector instead.
    """
    return [frame async for frame in async_gen]


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
        # Early-return path must still stop the TTFB timer exactly once
        # — Pipecat starts it in _push_tts_frames before calling run_tts,
        # so a missed stop leaves the metric stuck.
        assert svc.stop_ttfb_metrics.call_count == 1

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
        # TTFB timer must be stopped on the defensive early-return
        # path too — Pipecat starts it upstream before calling run_tts.
        assert svc.stop_ttfb_metrics.call_count == 1

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

        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        frames_by_ctx: dict[str, list[TTSAudioRawFrame]] = {}

        async def _race():
            async def _release_after_both_started():
                await asyncio.sleep(0.02)
                start_event.set()

            asyncio.get_event_loop().create_task(_release_after_both_started())

            async def _collect(gen):
                return [frame async for frame in gen]

            results = await asyncio.gather(
                _collect(svc_a.run_tts("a", context_id="ctx-a")),
                _collect(svc_b.run_tts("b", context_id="ctx-b")),
            )
            # Capture the frames so the assertions below can verify
            # both calls actually produced audio. Without this, a bug
            # that made both calls ErrorFrame before reaching the lock
            # would leave max_in_flight=0 and pass the serialization
            # check with a false sense of coverage.
            frames_by_ctx["a"] = results[0]
            frames_by_ctx["b"] = results[1]

        asyncio.run(_race())

        assert max_in_flight == 1, (
            f"expected serialized tts() calls, got max_in_flight={max_in_flight}"
        )
        # Both calls must have produced at least one audio frame; if either
        # bailed out with an ErrorFrame before the lock, max_in_flight would
        # still be <= 1 and the serialization check alone would pass.
        for ctx in ("a", "b"):
            audio_frames = [f for f in frames_by_ctx[ctx] if isinstance(f, TTSAudioRawFrame)]
            assert audio_frames, (
                f"ctx-{ctx} produced no audio frames — lock test is not actually exercising synth"
            )

    def test_next_synth_waits_for_previous_thread_exit(self, tmp_path):
        """Regression guard for the CancelledError race (issue #26).

        Scenario: the FIRST synth's coroutine is cancelled mid-
        ``to_thread(tts)``. The asyncio lock releases via ``__aexit__``,
        but the background threadpool worker keeps running until it
        naturally completes — at which point ``_run_xtts_synth_with_gate``'s
        try/finally sets ``_xtts_synth_idle``.

        The SECOND synth must NOT execute ``tts()`` concurrently with
        the first thread. It must wait on ``_xtts_synth_idle.wait()``
        until the first thread's finally runs. Without the backstop,
        the second synth would start immediately after the asyncio
        lock release and corrupt the shared ``nn.Module`` hidden state.

        We simulate cancellation by having the first tts() call sleep
        long enough that the coroutine gets cancelled while the thread
        is still running. The second synth's start time is compared
        against the first thread's actual exit time.
        """
        import threading  # noqa: PLC0415

        _make_profile(tmp_path)

        first_tts_exited = threading.Event()
        second_tts_entered = threading.Event()
        second_tts_entry_time = [0.0]
        first_exit_time = [0.0]

        def first_tts(**_kwargs):
            time.sleep(0.15)  # long enough that the coroutine gets cancelled
            first_exit_time[0] = time.monotonic()
            first_tts_exited.set()
            return [0.1] * 12000

        def second_tts(**_kwargs):
            second_tts_entry_time[0] = time.monotonic()
            second_tts_entered.set()
            return [0.2] * 12000

        svc_first, fake_first = _build_service(
            voice_library_dir=tmp_path,
            tts_return=None,
        )
        fake_first.tts = first_tts
        svc_second, fake_second = _build_service(
            voice_library_dir=tmp_path,
            tts_return=None,
        )
        fake_second.tts = second_tts

        async def _race():
            # Start the first synth and cancel it after ~30 ms (while
            # the thread is still sleeping).
            first_task = asyncio.create_task(
                _drain_async(svc_first.run_tts("first", context_id="ctx-1"))
            )
            await asyncio.sleep(0.03)
            first_task.cancel()
            try:
                await first_task
            except asyncio.CancelledError:
                pass

            # Now kick off the second synth. It must wait inside the
            # lock for _xtts_synth_idle before calling tts().
            await _drain_async(svc_second.run_tts("second", context_id="ctx-2"))

        asyncio.run(_race())

        # The second synth must have entered tts() AFTER the first
        # thread's finally set the idle event. If the backstop were
        # absent, the second synth would have started while the first
        # thread was still sleeping — entry_time < exit_time.
        assert second_tts_entered.is_set(), "second synth never ran"
        assert first_tts_exited.is_set(), "first synth thread never completed"
        assert second_tts_entry_time[0] >= first_exit_time[0], (
            f"second synth started at {second_tts_entry_time[0]:.4f} BEFORE first "
            f"thread exited at {first_exit_time[0]:.4f} — backstop didn't fire, "
            f"concurrent tts() calls on shared nn.Module"
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

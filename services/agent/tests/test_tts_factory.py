"""Tests for services/agent/tts_factory.build_tts_service.

PiperTTSService's constructor eagerly downloads the voice pack from
HuggingFace (see pipecat/services/piper/tts.py:106-108), so instantiating
it in a CPU-only test environment would hit the network. We monkey-patch
PiperTTSService at its import location to capture the factory's dispatch
without paying that cost. This mirrors the "stub heavy deps" pattern used
by test_gigaam_stt.py.

Convention: sync tests + asyncio.run() for async bodies. The agent test
suite deliberately avoids pytest-asyncio. No async code here yet, but
the import-ordering pattern (noqa: E402 after stub injection) matches
the existing tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline import AgentConfig  # noqa: E402 — imported at module scope; no heavy deps pulled through
from tts_factory import build_tts_service  # noqa: E402


def _make_cfg(*, tts_engine: str = "piper") -> AgentConfig:
    """Build a minimal AgentConfig for factory-dispatch tests."""
    return AgentConfig(
        livekit_url="ws://test",
        livekit_token="test-token",
        room_name="test-room",
        vllm_base_url="http://test",
        vllm_model="test-model",
        tts_voice="ru_RU-denis-medium",
        vllm_api_key="test-key",
        piper_voices_dir=Path("/tmp/piper-voices-test"),  # noqa: S108 — test-only sentinel
        tts_engine=tts_engine,
    )


# ─── Happy path ────────────────────────────────────────────────────────


class TestPiperBranch:
    def test_piper_dispatches_to_piper_service(self, monkeypatch):
        """Factory returns a PiperTTSService instance for engine=piper.

        PiperTTSService's real __init__ downloads the voice pack. We patch
        the class at its import location inside the factory so we can
        assert the dispatch happened with the right kwargs without
        touching the network.
        """
        fake_instance = MagicMock(name="PiperTTSService_instance")
        fake_cls = MagicMock(name="PiperTTSService_cls", return_value=fake_instance)
        # Settings is accessed as a class attribute before construction;
        # preserve that by attaching a trivial MagicMock.
        fake_cls.Settings = MagicMock(name="PiperTTSSettings")

        import pipecat.services.piper.tts as piper_mod  # noqa: PLC0415

        monkeypatch.setattr(piper_mod, "PiperTTSService", fake_cls)

        cfg = _make_cfg(tts_engine="piper")
        result = build_tts_service("piper", cfg=cfg, voice="ru_RU-denis-medium")

        assert result is fake_instance
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["download_dir"] == cfg.piper_voices_dir
        assert kwargs["use_cuda"] is False
        # Settings was invoked with the provided voice string. The factory
        # constructs Settings(voice=voice) — assert it was called with that
        # argument and that the result flowed into the service init.
        fake_cls.Settings.assert_called_once_with(voice="ru_RU-denis-medium")
        assert kwargs["settings"] is fake_cls.Settings.return_value

    def test_piper_passes_voice_through_verbatim(self, monkeypatch):
        """voice argument is provider-specific; factory does not translate."""
        fake_cls = MagicMock(name="PiperTTSService_cls")
        fake_cls.Settings = MagicMock(name="PiperTTSSettings")

        import pipecat.services.piper.tts as piper_mod  # noqa: PLC0415

        monkeypatch.setattr(piper_mod, "PiperTTSService", fake_cls)

        cfg = _make_cfg()
        build_tts_service("piper", cfg=cfg, voice="ru_RU-irina-medium")

        fake_cls.Settings.assert_called_once_with(voice="ru_RU-irina-medium")


# ─── Error path ────────────────────────────────────────────────────────


class TestUnknownEngine:
    def test_unknown_engine_raises_value_error(self):
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="unknown TTS_ENGINE"):
            build_tts_service("nope", cfg=cfg, voice="anything")

    def test_unknown_engine_name_appears_in_message(self):
        """Operators need to see the offending value in the error text."""
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="nope"):
            build_tts_service("nope", cfg=cfg, voice="anything")

    def test_empty_engine_name_raises(self):
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="unknown TTS_ENGINE"):
            build_tts_service("", cfg=cfg, voice="anything")


# ─── Placeholders for future units ─────────────────────────────────────


class TestSileroBranch:
    """Factory dispatch for engine=silero (Unit 3).

    SileroTTSService eagerly binds the preloaded torch.hub model. Since
    torch + silero aren't installed on the CPU-only CI runner, we stub
    ``get_silero`` at its import location inside the factory to hand back
    a MagicMock — matching the "stub heavy deps" pattern used by the
    piper branch tests above and the gigaam_stt tests.
    """

    def test_silero_dispatches_to_silero_service(self, monkeypatch):
        import models  # noqa: PLC0415

        fake_model = MagicMock(name="silero_model")
        monkeypatch.setattr(models, "get_silero", lambda: fake_model)

        cfg = _make_cfg(tts_engine="silero")
        result = build_tts_service("silero", cfg=cfg, voice="v5_cis_base")

        # Import the class after the factory has run, so the same module
        # object is shared (factory's lazy import caches it in sys.modules).
        from silero_tts import SileroTTSService  # noqa: PLC0415

        assert isinstance(result, SileroTTSService)
        # The preloaded singleton is injected into the service as-is.
        assert result._loaded_model is fake_model

    def test_silero_voice_flows_into_settings(self, monkeypatch):
        """voice string is passed through to SileroSettings.speaker."""
        import models  # noqa: PLC0415

        fake_model = MagicMock(name="silero_model")
        monkeypatch.setattr(models, "get_silero", lambda: fake_model)

        cfg = _make_cfg(tts_engine="silero")
        result = build_tts_service("silero", cfg=cfg, voice="v5_ru_custom")

        assert result._silero_settings.speaker == "v5_ru_custom"


@pytest.mark.skip(reason="added in Unit 4 — qwen3 sidecar + factory branch")
def test_qwen3_branch_returns_openai_tts_service():
    """Unit 4 un-skips this and asserts OpenAITTSService dispatch."""

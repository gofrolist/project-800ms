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


def _make_cfg(
    *,
    tts_engine: str = "piper",
    qwen3_base_url: str = "",
    qwen3_api_key: str = "",
) -> AgentConfig:
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
        qwen3_base_url=qwen3_base_url,
        qwen3_api_key=qwen3_api_key,
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
        """voice string is passed through to the TTSSettings store.

        SileroSettings is consumed eagerly in ``__init__`` (no stored
        reference after construction), so the assertion lives on the
        framework's ``_settings`` — the single source of truth after
        init.
        """
        import models  # noqa: PLC0415

        fake_model = MagicMock(name="silero_model")
        monkeypatch.setattr(models, "get_silero", lambda: fake_model)

        cfg = _make_cfg(tts_engine="silero")
        result = build_tts_service("silero", cfg=cfg, voice="v5_ru_custom")

        assert result._settings.model == "v5_ru_custom"
        assert result._settings.voice == "v5_ru_custom"


class TestQwen3Branch:
    """Factory dispatch for engine=qwen3 (Unit 4).

    Qwen3 routes through ``Qwen3TTSService`` — a thin subclass of
    Pipecat's upstream ``OpenAITTSService`` that adds network-error
    redaction (services/agent/qwen3_tts.py). The sidecar itself lives
    at infra/qwen3-tts-wrapper/. We patch ``Qwen3TTSService`` at its
    import location inside the factory (``qwen3_tts`` module) to
    capture the dispatch kwargs without touching openai-sdk
    construction or the network. Same mocking pattern as the piper
    branch tests above.
    """

    _VALID_QWEN3_KWARGS: dict[str, str] = {
        "qwen3_base_url": "http://qwen3-tts:8000/v1",
        "qwen3_api_key": "test-qwen3-secret",
    }

    def test_qwen3_dispatches_to_qwen3_tts_service(self, monkeypatch):
        """Factory returns a Qwen3TTSService with the expected base_url,
        api_key, model (tts-1-ru — wrapper LANGUAGE_CODE_MAPPING suffix for
        Russian), and a whitelisted voice."""
        fake_instance = MagicMock(name="Qwen3TTSService_instance")
        fake_cls = MagicMock(name="Qwen3TTSService_cls", return_value=fake_instance)

        import qwen3_tts as qwen3_mod  # noqa: PLC0415

        monkeypatch.setattr(qwen3_mod, "Qwen3TTSService", fake_cls)

        cfg = _make_cfg(tts_engine="qwen3", **self._VALID_QWEN3_KWARGS)
        result = build_tts_service("qwen3", cfg=cfg, voice="alloy")

        assert result is fake_instance
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["base_url"] == self._VALID_QWEN3_KWARGS["qwen3_base_url"]
        assert kwargs["api_key"] == self._VALID_QWEN3_KWARGS["qwen3_api_key"]
        # Russian routing is encoded in the model alias — the wrapper's
        # LANGUAGE_CODE_MAPPING strips the "-ru" suffix and selects Russian.
        assert kwargs["model"] == "tts-1-ru"
        # "alloy" is in Pipecat's VALID_VOICES whitelist, so it passes
        # through without substitution.
        assert kwargs["voice"] == "alloy"
        # Regression guard: the factory must pass a non-zero sample_rate.
        # Without it, Pipecat's TTSService.chunk_size property
        # (``sample_rate * 0.5 * 2``) evaluates to 0, httpx.iter_bytes(0)
        # yields no chunks, and the live pipeline plays silence with no
        # error frame. Observed on 2026-04-22 on the coastalai.ai deploy
        # — the sidecar logged successful generations while the browser
        # heard nothing. Must match the wrapper's native PCM output
        # rate (24 kHz; see DEFAULT_SAMPLE_RATE in
        # infra/qwen3-tts-wrapper/api/services/audio_encoding.py).
        assert kwargs["sample_rate"] == 24000

    def test_qwen3_missing_base_url_raises(self, monkeypatch):
        """Empty QWEN3_TTS_BASE_URL → ValueError that names only the
        missing field (not api_key).

        Each qwen3 field gets its own guard so the error message tells
        the operator exactly which env var to set. If the base_url guard
        is moved after the api_key guard, or collapsed back into a
        combined check, this test fails.

        The Qwen3TTSService mock should NOT be invoked in this case —
        assert that too so a future refactor that swaps the validation
        order can't silently construct a service with a bad URL.
        """
        fake_cls = MagicMock(name="Qwen3TTSService_cls")
        import qwen3_tts as qwen3_mod  # noqa: PLC0415

        monkeypatch.setattr(qwen3_mod, "Qwen3TTSService", fake_cls)

        cfg = _make_cfg(
            tts_engine="qwen3",
            qwen3_base_url="",
            qwen3_api_key="test-qwen3-secret",
        )
        with pytest.raises(ValueError) as exc_info:
            build_tts_service("qwen3", cfg=cfg, voice="alloy")

        message = str(exc_info.value)
        assert "QWEN3_TTS_BASE_URL" in message
        # Distinct per-field message must not mention the other field.
        assert "QWEN3_TTS_API_KEY" not in message
        fake_cls.assert_not_called()

    def test_qwen3_missing_api_key_raises(self, monkeypatch):
        """Empty QWEN3_TTS_API_KEY → ValueError that names only the
        missing field (not base_url).

        Symmetric to the base_url case — the per-field guard must not
        drag the other env name into an unrelated failure message.
        """
        fake_cls = MagicMock(name="Qwen3TTSService_cls")
        import qwen3_tts as qwen3_mod  # noqa: PLC0415

        monkeypatch.setattr(qwen3_mod, "Qwen3TTSService", fake_cls)

        cfg = _make_cfg(
            tts_engine="qwen3",
            qwen3_base_url="http://qwen3-tts:8000/v1",
            qwen3_api_key="",
        )
        with pytest.raises(ValueError) as exc_info:
            build_tts_service("qwen3", cfg=cfg, voice="alloy")

        message = str(exc_info.value)
        assert "QWEN3_TTS_API_KEY" in message
        # Distinct per-field message must not mention the other field.
        assert "QWEN3_TTS_BASE_URL" not in message
        fake_cls.assert_not_called()

    def test_qwen3_substitutes_non_whitelisted_voice(self, monkeypatch):
        """A Piper/Silero voice (e.g. "ru_RU-denis-medium") is not in
        Pipecat's OpenAITTSService whitelist — the factory substitutes
        "echo" (maps to Qwen3's Ryan voice in the wrapper) and logs a
        warning so the operator sees the mismatch.

        Loguru doesn't propagate to stdlib logging by default, so we
        attach a list-sink handler and assert on captured message text
        directly (instead of bridging into pytest's caplog, which would
        require wiring InterceptHandler at conftest scope).
        """
        fake_instance = MagicMock(name="Qwen3TTSService_instance")
        fake_cls = MagicMock(name="Qwen3TTSService_cls", return_value=fake_instance)

        import qwen3_tts as qwen3_mod  # noqa: PLC0415

        monkeypatch.setattr(qwen3_mod, "Qwen3TTSService", fake_cls)

        from loguru import logger  # noqa: PLC0415

        captured: list[str] = []
        handler_id = logger.add(
            lambda message: captured.append(message.record["message"]),
            level="WARNING",
            format="{message}",
        )
        try:
            cfg = _make_cfg(tts_engine="qwen3", **self._VALID_QWEN3_KWARGS)
            build_tts_service(
                "qwen3",
                cfg=cfg,
                voice="ru_RU-denis-medium",
            )
        finally:
            logger.remove(handler_id)

        kwargs = fake_cls.call_args.kwargs
        assert kwargs["voice"] == "echo"
        # The warning message names the offending voice + the substitution.
        # Loguru's format-spec parser does not honor the ``!r`` conversion
        # flag, so the factory precomputes ``repr(voice)`` and passes it
        # as a plain string kwarg. The captured message must therefore
        # contain the quoted reprs (e.g. "'ru_RU-denis-medium'") — not
        # the literal ``{voice!r}`` template and not the bare names.
        assert any(
            "'ru_RU-denis-medium'" in message and "'echo'" in message for message in captured
        ), (
            f"expected a substitution warning with quoted repr forms in "
            f"loguru output, got {captured!r}"
        )

    def test_qwen3_whitelisted_voice_passes_through(self, monkeypatch):
        """Every whitelisted OpenAI voice name flows through unchanged.

        Spot-checks a few voices to cover the happy path beyond "alloy"
        (which is the Pipecat default) — "echo" and "fable" are both
        mapped by the wrapper's voice table (echo→Ryan, fable→Serena).
        """
        fake_cls = MagicMock(name="Qwen3TTSService_cls")
        import qwen3_tts as qwen3_mod  # noqa: PLC0415

        monkeypatch.setattr(qwen3_mod, "Qwen3TTSService", fake_cls)

        cfg = _make_cfg(tts_engine="qwen3", **self._VALID_QWEN3_KWARGS)
        for voice in ("echo", "fable", "shimmer"):
            fake_cls.reset_mock()
            build_tts_service("qwen3", cfg=cfg, voice=voice)
            assert fake_cls.call_args.kwargs["voice"] == voice


# ─── AgentConfig.tts_engine validation ─────────────────────────────────


class TestAgentConfigTtsEngine:
    """__post_init__ validation fails at construction instead of at
    first build_task call.

    The factory still rejects unknown engine names at dispatch time
    (TestUnknownEngine), but catching the error at AgentConfig()
    surfaces the config bug on agent boot, not on the first room
    dispatch — which might happen minutes after deploy.
    """

    def test_unknown_engine_raises_at_construction(self):
        with pytest.raises(ValueError, match="tts_engine must be one of"):
            _make_cfg(tts_engine="unknown")

    def test_error_message_includes_offending_value(self):
        """Operators need to see what they typed in the error text."""
        with pytest.raises(ValueError, match="'banana'"):
            _make_cfg(tts_engine="banana")

    def test_error_message_includes_valid_options(self):
        """Operators need to see the valid options to correct the typo."""
        with pytest.raises(ValueError) as exc_info:
            _make_cfg(tts_engine="nope")
        message = str(exc_info.value)
        assert "piper" in message
        assert "silero" in message
        assert "qwen3" in message

    def test_empty_engine_name_raises_at_construction(self):
        with pytest.raises(ValueError, match="tts_engine must be one of"):
            _make_cfg(tts_engine="")

    def test_valid_engines_construct_cleanly(self):
        """Each supported engine name passes validation."""
        for engine in ("piper", "silero", "qwen3"):
            cfg = _make_cfg(tts_engine=engine)
            assert cfg.tts_engine == engine

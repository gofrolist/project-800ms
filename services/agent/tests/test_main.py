"""Tests for the agent entrypoint (main.py).

Scope: the conditional Silero preload and its hard-fail exit semantics.
The rest of ``main()`` (env collection, HTTP server wiring) is exercised
at integration level; here we assert the boot-time guard that was added
alongside the TTS abstraction: Silero MUST pre-load when TTS_ENGINE is
silero, and a preload failure MUST exit with code 3 so the container
restarts instead of serving traffic on a half-initialized agent.

Convention mirrors test_gigaam_stt / test_silero_tts: heavy deps
(``gigaam`` in particular — not pip-installable in CI) are stubbed at
module scope before importing the modules under test, and tests are
sync with ``asyncio.run()`` for async bodies where needed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Heavy deps — stub before importing main so lazy ``import gigaam`` /
# ``import torch`` inside ``models`` resolve to mocks.
sys.modules.setdefault("gigaam", MagicMock())

import main  # noqa: E402 — must import after stub injection
from models import _reset_models_for_tests  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset cached Silero/GigaAM singletons between tests and clear
    the module-level state ``main`` mutates so each test starts clean."""
    _reset_models_for_tests()
    # main() mutates module globals — reset them so cross-test state
    # (e.g. _livekit_url left from an earlier test) can't leak in.
    main._livekit_url = ""
    main._api_key = ""
    main._api_secret = ""
    main._base_config = {}
    main._active_rooms.clear()
    # Ensure env vars required by main() are present for these tests.
    monkeypatch.setenv("LIVEKIT_URL", "ws://test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-long-enough-32-chars-xyz")
    monkeypatch.setenv("VLLM_BASE_URL", "http://test:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "test-vllm")
    yield
    _reset_models_for_tests()


def _patch_runtime_bits():
    """Patch the heavy runtime surfaces main() touches so the test
    never actually binds a socket or loads a real model.

    Returns the set of MagicMock replacements for later assertions.
    """
    return {
        "load_gigaam": patch.object(main, "load_gigaam", MagicMock()),
        "load_silero": patch.object(main, "load_silero", MagicMock()),
        "load_xtts": patch.object(main, "load_xtts", MagicMock()),
        "web_run_app": patch.object(main.web, "run_app", MagicMock()),
    }


class TestSileroPreload:
    def test_preloads_silero_when_tts_engine_silero(self, monkeypatch):
        """With TTS_ENGINE=silero, main() must call load_silero() exactly
        once before handing off to the HTTP server.

        Asserts the conditional in main.py's preload block actually
        fires for the silero engine — a regression where someone
        collapses the preload into a Piper-only path would cause every
        first-dispatch synth to pay the ~200 MB torch.hub download.
        """
        monkeypatch.setenv("TTS_ENGINE", "silero")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"] as fake_gigaam,
            patches["load_silero"] as fake_silero,
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

        fake_gigaam.assert_called_once()
        fake_silero.assert_called_once()
        # XTTS must stay out of a silero-only deploy — its 1.8 GB
        # checkpoint download would dwarf the silero preload time.
        fake_xtts.assert_not_called()

    def test_skips_silero_preload_when_tts_engine_piper(self, monkeypatch):
        """With TTS_ENGINE=piper, load_silero must NOT be called.

        Piper/Qwen3 deploys don't need the Silero model and shouldn't
        pay its download/load cost. This test pins the behavior so a
        refactor can't accidentally preload Silero unconditionally.
        """
        monkeypatch.setenv("TTS_ENGINE", "piper")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"] as fake_gigaam,
            patches["load_silero"] as fake_silero,
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

        fake_gigaam.assert_called_once()
        fake_silero.assert_not_called()
        fake_xtts.assert_not_called()

    def test_exits_3_when_silero_preload_fails(self, monkeypatch):
        """A Silero load failure at startup must exit with code 3.

        We hard-fail instead of starting the HTTP server because an
        agent that boots but can't synthesize will silently drop every
        /dispatch into an ErrorFrame — worse than a restart loop.
        """
        monkeypatch.setenv("TTS_ENGINE", "silero")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"] as fake_silero,
            patches["load_xtts"],
            patches["web_run_app"],
        ):
            fake_silero.side_effect = RuntimeError("torch.hub download failed")

            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 3

    def test_exits_3_when_gigaam_preload_fails(self, monkeypatch):
        """Symmetric to the Silero case — GigaAM preload failures
        must also exit 3. Pins the existing hard-fail behavior so a
        regression in either branch surfaces here.
        """
        monkeypatch.setenv("TTS_ENGINE", "piper")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"] as fake_gigaam,
            patches["load_silero"],
            patches["load_xtts"],
            patches["web_run_app"],
        ):
            fake_gigaam.side_effect = RuntimeError("gigaam weights corrupt")

            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 3


class TestXttsPreload:
    """XTTS preload is gated on TTS_PRELOAD_ENGINES including ``xtts`` —
    same shape as the Silero case above, but XTTS downloads a ~1.8 GB
    checkpoint on cold cache so the distinction (preload vs lazy) matters
    even more than for Silero.
    """

    def test_preloads_xtts_when_tts_engine_xtts(self, monkeypatch):
        """With TTS_ENGINE=xtts, main() must call load_xtts() exactly
        once before handing off to the HTTP server."""
        monkeypatch.setenv("TTS_ENGINE", "xtts")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"] as fake_gigaam,
            patches["load_silero"] as fake_silero,
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

        fake_gigaam.assert_called_once()
        fake_xtts.assert_called_once()
        # An xtts-only deploy must not pay Silero's download cost.
        fake_silero.assert_not_called()

    def test_preloads_both_silero_and_xtts_when_listed(self, monkeypatch):
        """Demo-site config: TTS_PRELOAD_ENGINES=piper,silero,qwen3,xtts
        must preload both Silero and XTTS so the first session on either
        engine doesn't cold-start."""
        monkeypatch.setenv("TTS_ENGINE", "piper")
        monkeypatch.setenv("TTS_PRELOAD_ENGINES", "piper,silero,qwen3,xtts")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"] as fake_silero,
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

        fake_silero.assert_called_once()
        fake_xtts.assert_called_once()

    def test_skips_xtts_preload_when_tts_engine_piper(self, monkeypatch):
        """With TTS_ENGINE=piper and no XTTS in the preload list,
        load_xtts must NOT be called — XTTS's checkpoint is ~1.8 GB and
        must not be downloaded on a piper deploy."""
        monkeypatch.setenv("TTS_ENGINE", "piper")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

        fake_xtts.assert_not_called()

    def test_exits_3_when_xtts_preload_fails(self, monkeypatch):
        """An XTTS load failure at startup must exit with code 3 — same
        hard-fail semantics as the Silero case. Boot-but-can't-synth is
        strictly worse than a container restart loop.
        """
        monkeypatch.setenv("TTS_ENGINE", "xtts")
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            fake_xtts.side_effect = RuntimeError("CPML EULA not accepted")

            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 3

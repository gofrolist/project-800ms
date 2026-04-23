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

    @staticmethod
    def _seed_voice_library(monkeypatch, tmp_path):
        """Populate XTTS_VOICE_LIBRARY_DIR with a dummy profile so the
        startup check doesn't short-circuit before load_xtts is called."""
        monkeypatch.setenv("XTTS_VOICE_LIBRARY_DIR", str(tmp_path))
        profile = tmp_path / "profiles" / "demo-ru"
        profile.mkdir(parents=True)
        (profile / "meta.json").write_text("{}", encoding="utf-8")

    def test_preloads_xtts_when_tts_engine_xtts(self, monkeypatch, tmp_path):
        """With TTS_ENGINE=xtts, main() must call load_xtts() exactly
        once before handing off to the HTTP server."""
        monkeypatch.setenv("TTS_ENGINE", "xtts")
        self._seed_voice_library(monkeypatch, tmp_path)
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

    def test_preloads_both_silero_and_xtts_when_listed(self, monkeypatch, tmp_path):
        """Demo-site config: TTS_PRELOAD_ENGINES=piper,silero,qwen3,xtts
        must preload both Silero and XTTS so the first session on either
        engine doesn't cold-start."""
        monkeypatch.setenv("TTS_ENGINE", "piper")
        monkeypatch.setenv("TTS_PRELOAD_ENGINES", "piper,silero,qwen3,xtts")
        self._seed_voice_library(monkeypatch, tmp_path)
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

        Exception here is a plain RuntimeError (not OSError/ENOSPC), so
        the graceful-degrade branch does NOT apply — we want the hard
        fail for code/dep bugs that won't self-heal on restart.
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

    def test_exits_3_when_voice_library_missing(self, monkeypatch, tmp_path):
        """XTTS preload with a missing voice_library mount must hard-fail
        at boot rather than limping along and silently ErrorFrame-ing on
        every dispatch. Docker compose silently auto-creates the mount
        target if the host path is absent, so the first dispatch would
        otherwise produce ``_resolve_voice_profile`` ValueError → swallowed
        → dead session.
        """
        monkeypatch.setenv("TTS_ENGINE", "xtts")
        # Point at a valid tmp dir that's missing the profiles/ subdir
        # (or has an empty one) — the startup check should refuse both.
        monkeypatch.setenv("XTTS_VOICE_LIBRARY_DIR", str(tmp_path))
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 3
            # load_xtts must NOT be invoked — the startup check fires
            # first and short-circuits.
            fake_xtts.assert_not_called()

    def test_boots_when_voice_library_has_profile(self, monkeypatch, tmp_path):
        """Positive case: a voice_library_dir containing at least one
        profile lets XTTS preload proceed to load_xtts."""
        monkeypatch.setenv("TTS_ENGINE", "xtts")
        monkeypatch.setenv("XTTS_VOICE_LIBRARY_DIR", str(tmp_path))
        profile = tmp_path / "profiles" / "demo-ru"
        profile.mkdir(parents=True)
        (profile / "meta.json").write_text("{}", encoding="utf-8")

        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

            fake_xtts.assert_called_once()


class TestXttsDiskSpaceDegrade:
    """Graceful degrade: insufficient free disk at TTS_HOME must skip
    the XTTS preload rather than sys.exit(3) the whole agent. Other
    engines (Piper, Silero, Qwen3) keep serving. handle_dispatch's 409
    for absent engines then covers sessions that still request xtts.
    """

    def _seed_voice_library(self, monkeypatch, tmp_path):
        profile = tmp_path / "profiles" / "demo-ru"
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "meta.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("XTTS_VOICE_LIBRARY_DIR", str(tmp_path))

    def test_low_disk_skips_preload_and_drops_xtts_from_preload_set(self, monkeypatch, tmp_path):
        """With <2 GB free at TTS_HOME, agent boots normally but xtts
        is NOT in _preload_engines. load_xtts is never called.
        """
        import shutil as real_shutil  # noqa: PLC0415

        monkeypatch.setenv("TTS_ENGINE", "xtts")
        self._seed_voice_library(monkeypatch, tmp_path)

        # Simulate a near-full volume — 500 MB free, below the 2 GB threshold.
        def _fake_disk_usage(path):
            return real_shutil._ntuple_diskusage(
                total=10_000_000_000,
                used=9_500_000_000,
                free=500_000_000,
            )

        monkeypatch.setattr(main.shutil, "disk_usage", _fake_disk_usage)
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

            # Preload NOT attempted.
            fake_xtts.assert_not_called()
            # _preload_engines does NOT contain xtts — dispatch will 409.
            assert "xtts" not in main._preload_engines

    def test_enospc_mid_download_drops_xtts_not_exit_3(self, monkeypatch, tmp_path):
        """Pre-check passes (enough free), but the actual download
        raises ENOSPC (something else consumed space in the meantime).
        Must still degrade gracefully, not sys.exit(3).
        """
        import shutil as real_shutil  # noqa: PLC0415

        monkeypatch.setenv("TTS_ENGINE", "xtts")
        self._seed_voice_library(monkeypatch, tmp_path)

        # Enough free at pre-check time.
        def _fake_disk_usage(path):
            return real_shutil._ntuple_diskusage(
                total=100_000_000_000,
                used=10_000_000_000,
                free=90_000_000_000,
            )

        monkeypatch.setattr(main.shutil, "disk_usage", _fake_disk_usage)
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            fake_xtts.side_effect = OSError(28, "No space left on device")  # errno 28 = ENOSPC

            # Must NOT raise SystemExit — this is the graceful path.
            main.main()

            fake_xtts.assert_called_once()
            # xtts dropped from preload set after the failed load.
            assert "xtts" not in main._preload_engines

    def test_non_enospc_oserror_still_exits_3(self, monkeypatch, tmp_path):
        """A non-ENOSPC OSError (permission denied, I/O error, etc.) is
        not a resource-constraint issue and should keep the existing
        hard-fail behavior — graceful-degrade is only for disk-full.
        """
        import shutil as real_shutil  # noqa: PLC0415

        monkeypatch.setenv("TTS_ENGINE", "xtts")
        self._seed_voice_library(monkeypatch, tmp_path)
        monkeypatch.setattr(
            main.shutil,
            "disk_usage",
            lambda _p: real_shutil._ntuple_diskusage(
                total=100_000_000_000, used=10_000_000_000, free=90_000_000_000
            ),
        )
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            fake_xtts.side_effect = OSError(13, "Permission denied")  # errno 13 = EACCES

            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 3

    def test_ample_disk_proceeds_to_preload(self, monkeypatch, tmp_path):
        """Baseline regression guard: when disk has plenty of space,
        the degrade path is not entered — load_xtts runs and xtts stays
        in the preload set.
        """
        import shutil as real_shutil  # noqa: PLC0415

        monkeypatch.setenv("TTS_ENGINE", "xtts")
        self._seed_voice_library(monkeypatch, tmp_path)
        monkeypatch.setattr(
            main.shutil,
            "disk_usage",
            lambda _p: real_shutil._ntuple_diskusage(
                total=100_000_000_000, used=10_000_000_000, free=90_000_000_000
            ),
        )
        patches = _patch_runtime_bits()
        with (
            patches["load_gigaam"],
            patches["load_silero"],
            patches["load_xtts"] as fake_xtts,
            patches["web_run_app"],
        ):
            main.main()

            fake_xtts.assert_called_once()
            assert "xtts" in main._preload_engines


class TestHandleDispatchEngineGuard:
    """handle_dispatch must refuse sessions requesting an engine that
    isn't in TTS_PRELOAD_ENGINES. Without this guard, dispatch returns
    200 + LiveKit token, pipeline failure inside _run_pipeline is
    swallowed, and the caller hears silence.
    """

    def _make_request(self, body):
        """Build a minimal aiohttp.web.Request stub that handle_dispatch can use.

        handle_dispatch reads only ``request.json()`` so the stub just
        needs that coroutine — the rest of the Request interface isn't
        exercised.
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        req = MagicMock()
        req.json = AsyncMock(return_value=body)
        return req

    def test_rejects_engine_not_in_preload_set(self, monkeypatch):
        # Simulate a Piper-only deploy.
        main._preload_engines = frozenset({"piper"})
        main._base_config = {"tts_engine": "piper"}
        main._active_rooms.clear()
        req = self._make_request({"room": "r1", "tts_engine": "xtts"})

        import asyncio  # noqa: PLC0415

        resp = asyncio.run(main.handle_dispatch(req))

        assert resp.status == 409
        # No room slot claimed.
        assert "r1" not in main._active_rooms

    def test_accepts_engine_in_preload_set(self, monkeypatch):
        """A dispatch for an engine listed in TTS_PRELOAD_ENGINES returns
        202-style 200 with "dispatched" status.
        """
        main._preload_engines = frozenset({"piper", "xtts"})
        main._base_config = {"tts_engine": "piper"}
        main._active_rooms.clear()

        # Patch _run_pipeline so the test doesn't actually spawn a
        # pipeline coroutine — we only care about the synchronous
        # dispatch decision path.
        async def _noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(main, "_run_pipeline", _noop)

        req = self._make_request({"room": "r2", "tts_engine": "xtts"})

        import asyncio  # noqa: PLC0415

        resp = asyncio.run(main.handle_dispatch(req))

        assert resp.status == 200
        assert "r2" in main._active_rooms

    def test_falls_back_to_base_engine_when_override_missing(self, monkeypatch):
        """No tts_engine in the body → fall back to TTS_ENGINE. If that
        engine is in the preload set, accept. Preserves legacy single-engine
        deploy behaviour.
        """
        main._preload_engines = frozenset({"silero"})
        main._base_config = {"tts_engine": "silero"}
        main._active_rooms.clear()

        async def _noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(main, "_run_pipeline", _noop)

        req = self._make_request({"room": "r3"})

        import asyncio  # noqa: PLC0415

        resp = asyncio.run(main.handle_dispatch(req))

        assert resp.status == 200

    def test_skips_engine_check_when_preload_set_empty(self, monkeypatch):
        """If the preload set is empty (shouldn't happen in practice —
        TTS_PRELOAD_ENGINES defaults to TTS_ENGINE), do not block. Avoids
        a regression where test setups that forget to seed _preload_engines
        would spuriously 409.
        """
        main._preload_engines = frozenset()
        main._base_config = {"tts_engine": ""}
        main._active_rooms.clear()

        async def _noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(main, "_run_pipeline", _noop)

        req = self._make_request({"room": "r4", "tts_engine": "xtts"})

        import asyncio  # noqa: PLC0415

        resp = asyncio.run(main.handle_dispatch(req))

        assert resp.status == 200

"""Tests for Qwen3TTSService (services/agent/qwen3_tts.py).

The subclass exists to translate httpx network-class exceptions raised
from Pipecat's upstream ``OpenAITTSService.run_tts`` into redacted
``ErrorFrame``s, so a sidecar outage doesn't silently tear down the
whole Pipecat session. Tests assert that translation happens — and
that non-network exceptions fall through untouched so Pipecat's own
error handling (``BadRequestError`` → ``ErrorFrame``) still runs.

Convention mirrors the other agent tests: sync bodies with
``asyncio.run`` for async generators, heavy deps stubbed at module
scope where needed.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

# gigaam lives alongside silero in models.py; stub it so shared imports
# don't trip when this test module happens to load first in CI.
sys.modules.setdefault("gigaam", MagicMock())

from qwen3_tts import Qwen3TTSService  # noqa: E402 — after stub injection


async def _drain(async_gen):
    """Collect frames from an AsyncGenerator into a list."""
    return [frame async for frame in async_gen]


def _make_service():
    """Build a Qwen3TTSService with only the minimal state tests need.

    ``OpenAITTSService.__init__`` constructs an openai-sdk client at
    init time; we skip that entirely because our tests patch the
    parent's ``run_tts`` method directly. Constructing via
    ``object.__new__`` sidesteps the init cost while still giving us
    a bound instance of the real class.
    """
    return Qwen3TTSService.__new__(Qwen3TTSService)


class TestQwen3NetworkErrorRedaction:
    def test_httpx_connect_error_yields_redacted_error_frame(self):
        """An httpx.ConnectError from the sidecar must become an
        ErrorFrame with a generic message — NOT propagate and tear
        down the pipeline.

        Without this subclass, Pipecat's upstream OpenAITTSService
        only catches openai's BadRequestError, so a sidecar unreach
        would bubble out of run_tts and the LiveKit session would
        drop silently.
        """
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()

        async def _boom(self, _text, _ctx):
            raise httpx.ConnectError("connection refused")
            yield  # pragma: no cover - never reached

        with patch(
            "qwen3_tts.OpenAITTSService.run_tts",
            new=_boom,
        ):
            frames = asyncio.run(_drain(svc.run_tts("привет", "ctx-1")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        # Internal detail (connection message) must not leak to the
        # client-visible frame — only the redacted generic message.
        assert frames[0].error == "TTS synth failed"
        assert "connection refused" not in frames[0].error

    def test_read_timeout_yields_redacted_error_frame(self):
        """httpx.ReadTimeout (slow sidecar) is also caught and
        redacted. A stuck backend would otherwise hang the pipeline
        until the openai-sdk's own timeout triggers, then bubble the
        exception."""
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()

        async def _timeout(self, _text, _ctx):
            raise httpx.ReadTimeout("read timed out")
            yield  # pragma: no cover

        with patch(
            "qwen3_tts.OpenAITTSService.run_tts",
            new=_timeout,
        ):
            frames = asyncio.run(_drain(svc.run_tts("текст", "ctx-2")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert frames[0].error == "TTS synth failed"

    def test_happy_path_frames_pass_through_unchanged(self):
        """When the sidecar responds normally, every frame yielded
        by the upstream ``run_tts`` reaches the caller as-is. The
        subclass must not swallow or duplicate frames.
        """
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        expected = [
            TTSAudioRawFrame(audio=b"\x00" * 48, sample_rate=24000, num_channels=1),
            TTSAudioRawFrame(audio=b"\x01" * 48, sample_rate=24000, num_channels=1),
        ]

        svc = _make_service()

        async def _ok(self, _text, _ctx):
            for frame in expected:
                yield frame

        with patch(
            "qwen3_tts.OpenAITTSService.run_tts",
            new=_ok,
        ):
            frames = asyncio.run(_drain(svc.run_tts("ok", "ctx-ok")))

        assert frames == expected

    def test_non_network_exception_propagates(self):
        """Exceptions that aren't in the network whitelist must
        propagate — Pipecat's upstream handling (BadRequestError →
        ErrorFrame, for example) still needs to run on 4xx/5xx
        responses; we don't want to over-catch and turn a 400
        "invalid voice" into a generic synth failure.
        """
        svc = _make_service()

        async def _bad(self, _text, _ctx):
            raise RuntimeError("something unrelated")
            yield  # pragma: no cover

        with patch(
            "qwen3_tts.OpenAITTSService.run_tts",
            new=_bad,
        ):
            with pytest.raises(RuntimeError, match="something unrelated"):
                asyncio.run(_drain(svc.run_tts("x", "ctx")))

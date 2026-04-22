"""Tests for Qwen3TTSService (services/agent/qwen3_tts.py).

The subclass re-implements ``run_tts`` (no ``super().run_tts()`` call)
to:

1. Inject ``extra_body={"stream": True}`` on the audio.speech request so
   the Qwen3-TTS sidecar yields PCM chunks as they're generated instead
   of buffering the full utterance. This is the critical latency fix
   that makes TTFB ~1-2s instead of ~20s on the L4.
2. Translate ``httpx`` network-class exceptions into redacted
   ``ErrorFrame``s so a sidecar outage doesn't tear down the pipeline.

Tests assert both behaviours.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

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
    """Build a Qwen3TTSService skipping openai-sdk client init."""
    svc = Qwen3TTSService.__new__(Qwen3TTSService)
    # Pipecat's FrameProcessor.__str__ reads _name; normally set by
    # __init__. Bypass with a sentinel so log lines don't crash.
    svc._name = "Qwen3TTSService#0"
    # Attach the minimal state that the rewritten run_tts reads.
    settings = MagicMock()
    settings.voice = "echo"
    settings.model = "tts-1-ru"
    settings.instructions = None
    settings.speed = None
    svc._settings = settings
    # Mock the pipecat metrics + sample_rate properties that run_tts
    # calls into. sample_rate=24000 mirrors what the factory injects.
    svc._sample_rate = 24000
    svc.start_tts_usage_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    return svc


def _mock_streaming_response(*, status_code=200, body_chunks=(b"\x00" * 48,)):
    """Build an AsyncMock that matches
    ``client.audio.speech.with_streaming_response.create(...)``'s async
    context-manager protocol, returning a response whose iter_bytes
    yields the provided body chunks.
    """
    response = MagicMock()
    response.status_code = status_code
    response.text = AsyncMock(return_value="")

    async def _iter(_chunk_size):
        for chunk in body_chunks:
            yield chunk

    response.iter_bytes = MagicMock(side_effect=lambda chunk_size: _iter(chunk_size))

    cm = AsyncMock()
    cm.__aenter__.return_value = response
    cm.__aexit__.return_value = None
    return response, cm


def _attach_client(svc, cm):
    """Stitch a mocked async context manager onto
    ``svc._client.audio.speech.with_streaming_response.create(**kwargs)``."""
    create = MagicMock(return_value=cm)
    client = MagicMock()
    client.audio.speech.with_streaming_response.create = create
    svc._client = client
    return create


class TestQwen3StreamingBody:
    def test_stream_true_is_sent_in_request_body(self):
        """The critical latency fix: extra_body={"stream": True} must
        reach the sidecar. Without it, the sidecar buffers the full
        utterance (observed TTFB ~20s on L4 with 1.7B-CustomVoice)
        instead of streaming as generated."""
        svc = _make_service()
        _response, cm = _mock_streaming_response(body_chunks=(b"\x00" * 48,))
        create = _attach_client(svc, cm)

        asyncio.run(_drain(svc.run_tts("привет", "ctx-1")))

        create.assert_called_once()
        kwargs = create.call_args.kwargs
        assert kwargs.get("extra_body") == {"stream": True}

    def test_openai_request_params_flow_through(self):
        """Beyond stream=True, the standard OpenAI speech params
        (input, model, voice, response_format) must also arrive in the
        request body. The wrapper interprets ``model="tts-1-ru"`` as
        the Russian routing alias."""
        svc = _make_service()
        svc._settings.voice = "alloy"
        svc._settings.model = "tts-1-ru"
        _response, cm = _mock_streaming_response(body_chunks=(b"\x00" * 48,))
        create = _attach_client(svc, cm)

        asyncio.run(_drain(svc.run_tts("текст", "ctx-2")))

        kwargs = create.call_args.kwargs
        assert kwargs["input"] == "текст"
        assert kwargs["model"] == "tts-1-ru"
        # VALID_VOICES is identity-mapped for OpenAI's own voice names;
        # assert the whitelist path runs rather than bypasses.
        assert kwargs["voice"] == "alloy"
        assert kwargs["response_format"] == "pcm"

    def test_body_chunks_yield_tts_audio_raw_frames(self):
        """Each non-empty chunk returned by r.iter_bytes must produce
        a TTSAudioRawFrame carrying the service's sample_rate."""
        from pipecat.frames.frames import TTSAudioRawFrame  # noqa: PLC0415

        svc = _make_service()
        chunks = (b"\x11" * 48, b"\x22" * 48, b"\x33" * 48)
        _response, cm = _mock_streaming_response(body_chunks=chunks)
        _attach_client(svc, cm)

        frames = asyncio.run(_drain(svc.run_tts("ok", "ctx-ok")))

        assert len(frames) == 3
        for frame, expected in zip(frames, chunks, strict=True):
            assert isinstance(frame, TTSAudioRawFrame)
            assert frame.audio == expected
            assert frame.sample_rate == 24000
            assert frame.num_channels == 1

    def test_empty_chunks_are_skipped(self):
        """Some wrappers emit a zero-byte chunk to signal EOF before
        the actual close — the loop must not yield an empty audio
        frame that LiveKit would reject."""
        svc = _make_service()
        _response, cm = _mock_streaming_response(body_chunks=(b"", b"\x11" * 24, b""))
        _attach_client(svc, cm)

        frames = asyncio.run(_drain(svc.run_tts("x", "c")))

        assert len(frames) == 1
        assert frames[0].audio == b"\x11" * 24


class TestQwen3ErrorResponses:
    def test_non_200_status_yields_error_frame(self):
        """A 4xx/5xx from the sidecar must become an ErrorFrame that
        carries the status + body text, not a silent drop."""
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()
        response, cm = _mock_streaming_response(status_code=500)
        response.text = AsyncMock(return_value="sidecar exploded")
        _attach_client(svc, cm)

        frames = asyncio.run(_drain(svc.run_tts("x", "c")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert "500" in frames[0].error
        assert "sidecar exploded" in frames[0].error


class TestQwen3NetworkErrorRedaction:
    def test_httpx_connect_error_yields_redacted_error_frame(self):
        """An httpx.ConnectError from the sidecar must become an
        ErrorFrame with a generic message — NOT propagate and tear
        down the pipeline.

        With streaming, the error most likely fires inside the
        ``with_streaming_response.create`` context manager or
        ``iter_bytes``. Patch ``create`` to raise before returning a
        context manager to exercise that path.
        """
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()
        client = MagicMock()
        client.audio.speech.with_streaming_response.create = MagicMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        svc._client = client

        frames = asyncio.run(_drain(svc.run_tts("привет", "ctx-1")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        # Internal detail must not leak to the client-visible frame —
        # only the redacted generic message.
        assert frames[0].error == "TTS synth failed"
        assert "connection refused" not in frames[0].error

    def test_read_timeout_yields_redacted_error_frame(self):
        """httpx.ReadTimeout during streamed-body consumption is also
        caught and redacted. A stuck backend mid-stream would otherwise
        bubble into the pipeline."""
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()
        client = MagicMock()
        client.audio.speech.with_streaming_response.create = MagicMock(
            side_effect=httpx.ReadTimeout("read timed out")
        )
        svc._client = client

        frames = asyncio.run(_drain(svc.run_tts("текст", "ctx-2")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert frames[0].error == "TTS synth failed"

    def test_remote_protocol_error_yields_redacted_error_frame(self):
        """httpx.RemoteProtocolError happens when the sidecar closes
        the connection mid-stream. Same redaction rule."""
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        svc = _make_service()
        client = MagicMock()
        client.audio.speech.with_streaming_response.create = MagicMock(
            side_effect=httpx.RemoteProtocolError("connection closed")
        )
        svc._client = client

        frames = asyncio.run(_drain(svc.run_tts("x", "c")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)
        assert frames[0].error == "TTS synth failed"


class TestQwen3ExceptionPassthrough:
    def test_bad_request_error_yields_error_frame(self):
        """openai's BadRequestError (e.g. unknown voice, bad model
        alias) must be turned into an ErrorFrame for Pipecat to
        surface. The subclass reuses the upstream mapping rather than
        propagating — matches pipecat/services/openai/tts.py behavior.
        """
        from pipecat.frames.frames import ErrorFrame  # noqa: PLC0415

        # openai.BadRequestError's __init__ requires a response object.
        # Construct a lightweight fake via the exception's message
        # parent class, bypassing the sdk's builder.
        from openai import BadRequestError  # noqa: PLC0415

        bad = BadRequestError.__new__(BadRequestError)
        BaseException.__init__(bad, "invalid voice 'banana'")

        svc = _make_service()
        client = MagicMock()
        client.audio.speech.with_streaming_response.create = MagicMock(side_effect=bad)
        svc._client = client

        frames = asyncio.run(_drain(svc.run_tts("x", "c")))

        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)

    def test_non_network_exception_propagates(self):
        """Exceptions that aren't in the network whitelist and aren't
        BadRequestError must propagate — a RuntimeError from our own
        wiring should not be silently redacted.
        """
        svc = _make_service()
        client = MagicMock()
        client.audio.speech.with_streaming_response.create = MagicMock(
            side_effect=RuntimeError("something unrelated")
        )
        svc._client = client

        with pytest.raises(RuntimeError, match="something unrelated"):
            asyncio.run(_drain(svc.run_tts("x", "c")))


class TestQwen3Metrics:
    def test_ttfb_and_usage_metrics_fire_on_happy_path(self):
        """Metrics hooks must run: start_tts_usage_metrics once per
        request with the input text; stop_ttfb_metrics on every yielded
        chunk (the upstream pattern — chunks are tiny so the extra
        awaits are cheap)."""
        svc = _make_service()
        _response, cm = _mock_streaming_response(
            body_chunks=(b"\x11" * 24, b"\x22" * 24),
        )
        _attach_client(svc, cm)

        asyncio.run(_drain(svc.run_tts("hi", "ctx")))

        svc.start_tts_usage_metrics.assert_awaited_once_with("hi")
        assert svc.stop_ttfb_metrics.await_count == 2

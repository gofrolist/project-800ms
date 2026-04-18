"""Tests for the TranscriptSink (no pipecat / CUDA imports)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


from transcript_sink import TranscriptSink


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestTranscriptSink:
    def test_empty_config_skips_silently(self) -> None:
        """Missing url or token → no HTTP call, no exception."""
        sink = TranscriptSink(api_base_url="", internal_token="", room="room-x")
        # Should not raise, should not hang. Timing a direct coroutine is
        # simplest via asyncio.run.
        asyncio.run(sink.log("user", "hi"))

    def test_posts_to_configured_url(self) -> None:
        """Happy path: sink hits the right URL with the right headers."""
        sink = TranscriptSink(
            api_base_url="http://api:8000",
            internal_token="secret-xxx",
            room="room-42",
        )

        mock_session_cm = AsyncMock()
        mock_post_cm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_post_cm.__aenter__.return_value = mock_resp
        mock_post_cm.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with patch("transcript_sink.aiohttp.ClientSession", return_value=mock_session_cm):
            asyncio.run(sink.log("assistant", "hello"))

        mock_session.post.assert_called_once()
        args, kwargs = mock_session.post.call_args
        assert args[0] == "http://api:8000/internal/transcripts"
        assert kwargs["headers"]["X-Internal-Token"] == "secret-xxx"
        assert kwargs["json"]["room"] == "room-42"
        assert kwargs["json"]["role"] == "assistant"
        assert kwargs["json"]["text"] == "hello"

    def test_non_2xx_logs_but_does_not_raise(self) -> None:
        """A 4xx/5xx from the API must NOT propagate into the pipeline."""
        sink = TranscriptSink(
            api_base_url="http://api:8000",
            internal_token="secret",
            room="r",
        )

        mock_session_cm = AsyncMock()
        mock_post_cm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="boom")
        mock_post_cm.__aenter__.return_value = mock_resp
        mock_post_cm.__aexit__.return_value = None
        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with patch("transcript_sink.aiohttp.ClientSession", return_value=mock_session_cm):
            # Should complete normally.
            asyncio.run(sink.log("user", "x"))

    def test_network_error_is_swallowed(self) -> None:
        """Connection refused / DNS glitch / timeout — also contained."""
        sink = TranscriptSink(
            api_base_url="http://api:8000",
            internal_token="secret",
            room="r",
        )

        with patch(
            "transcript_sink.aiohttp.ClientSession",
            side_effect=ConnectionError("boom"),
        ):
            asyncio.run(sink.log("user", "x"))

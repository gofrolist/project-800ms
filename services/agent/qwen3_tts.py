"""Network-redacting wrapper around Pipecat's ``OpenAITTSService``.

Pipecat's upstream ``OpenAITTSService.run_tts`` only catches the
openai-sdk ``BadRequestError`` (and re-yields it as an ``ErrorFrame``).
Every other exception class — notably ``httpx`` network errors raised
when the vendored Qwen3-TTS sidecar is slow / down / partitioned from
the agent's docker network — propagates out of the async generator.
Pipecat's pipeline then tears the whole session down without emitting a
caller-visible frame, which a LiveKit client sees as a silent
disconnect.

This thin subclass catches the network-class exceptions, logs the
internal cause server-side, and yields a redacted ``ErrorFrame`` so the
pipeline (and downstream transport) treats the failure the same way
``GigaAMSTTService`` / ``SileroTTSService`` treat their own synth
failures. Behavior mirrors ``gigaam_stt.py:171-177`` and
``silero_tts.py:217-223``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.openai.tts import OpenAITTSService


# Network-class httpx exceptions that must not tear down the pipeline.
# Listed explicitly (not ``httpx.HTTPError`` broadly) so that 4xx/5xx
# ``HTTPStatusError`` responses still reach Pipecat's own error handler
# — those signal a server-understood problem (bad voice name, bad
# model alias) that benefit from the upstream ``BadRequestError``
# translation path.
_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


class Qwen3TTSService(OpenAITTSService):
    """``OpenAITTSService`` with network-error redaction for the vendored
    Qwen3-TTS sidecar.

    Only overrides ``run_tts`` — every other aspect (voice whitelisting,
    model routing, TTFB metrics, frame shape) inherits from the upstream
    implementation. Keeps the subclass surface area tiny so future
    Pipecat bumps don't drift us into a maintenance fork.
    """

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        except _NETWORK_EXCEPTIONS:
            # Log the full exception server-side (includes the target
            # URL and the httpx error detail); emit a generic string
            # downstream so nothing about the sidecar's deploy shape
            # leaks into the LiveKit data channel.
            logger.exception("Qwen3 sidecar request failed")
            yield ErrorFrame("TTS synth failed")

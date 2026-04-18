"""Frame processors that forward transcript data to the web UI via
LiveKit data channel, and optionally persist to the API's /internal/
transcripts endpoint."""

from __future__ import annotations

import asyncio
import json

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.livekit.transport import LiveKitTransport

from transcript_sink import TranscriptSink

# Debounce delay: flush buffered user transcript after this many seconds
# of no new transcriptions.
_USER_TRANSCRIPT_DEBOUNCE_SECS = 1.0


async def _forward(
    transport: LiveKitTransport,
    sink: TranscriptSink | None,
    role: str,
    text: str,
) -> None:
    """Emit one transcript to both destinations.

    1. LiveKit data channel — renders in the web UI in real time.
    2. API internal endpoint — durable storage for playback / auditing.

    Both paths swallow their own errors (best-effort). A LiveKit-side
    failure shouldn't drop the DB write and vice versa.
    """
    msg = json.dumps({"role": role, "text": text}, ensure_ascii=False)
    logger.debug("Transcript role={role} chars={n}", role=role, n=len(text))
    try:
        await transport.send_message(msg)
    except Exception:
        logger.exception("Failed to send transcript over LiveKit data channel")
    if sink is not None:
        await sink.log(role, text)


class UserTranscriptForwarder(FrameProcessor):
    """Place between STT and user_agg to capture user transcriptions
    before user_agg consumes them.

    Buffers fragments and flushes them as a single message after a
    debounce period of silence, so that VAD-split words appear as one
    message in the UI.
    """

    def __init__(
        self,
        transport: LiveKitTransport,
        sink: TranscriptSink | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transport = transport
        self._sink = sink
        self._buf: list[str] = []
        self._flush_handle: asyncio.TimerHandle | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._buf.append(frame.text.strip())
            self._schedule_flush()
        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------

    def _schedule_flush(self):
        """Reset the debounce timer."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        loop = asyncio.get_running_loop()
        self._flush_handle = loop.call_later(
            _USER_TRANSCRIPT_DEBOUNCE_SECS,
            lambda: loop.create_task(self._flush()),
        )

    async def _flush(self):
        """Send the accumulated transcript fragments as one message."""
        self._flush_handle = None
        buf, self._buf = self._buf, []
        if not buf:
            return
        await _forward(self._transport, self._sink, "user", " ".join(buf))


class AssistantTranscriptForwarder(FrameProcessor):
    """Place after LLM to capture the full assistant response.
    Buffers streaming TextFrames between LLMFullResponseStart/End."""

    def __init__(
        self,
        transport: LiveKitTransport,
        sink: TranscriptSink | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transport = transport
        self._sink = sink
        self._buf: list[str] = []
        self._in_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._buf.clear()
        elif isinstance(frame, TextFrame) and self._in_response:
            self._buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            full = "".join(self._buf).strip()
            if full:
                await _forward(self._transport, self._sink, "assistant", full)
            self._buf.clear()

        await self.push_frame(frame, direction)

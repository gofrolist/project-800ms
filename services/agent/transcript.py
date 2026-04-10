"""Frame processors that forward transcript data to the web UI via LiveKit data channel."""

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
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.livekit.transport import LiveKitTransport

# Debounce delay: flush buffered user transcript after this many seconds
# of no new transcriptions.
_USER_TRANSCRIPT_DEBOUNCE_SECS = 1.0


async def _send_transcript(transport: LiveKitTransport, role: str, text: str):
    msg = json.dumps({"role": role, "text": text}, ensure_ascii=False)
    logger.debug("Transcript → client: {msg}", msg=msg)
    await transport.send_message(msg)


class UserTranscriptForwarder(FrameProcessor):
    """Place between STT and user_agg to capture user transcriptions
    before user_agg consumes them.

    Buffers fragments and flushes them as a single message after a
    debounce period of silence, so that VAD-split words appear as one
    message in the UI.
    """

    def __init__(self, transport: LiveKitTransport, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport
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
            lambda: asyncio.ensure_future(self._flush()),
        )

    async def _flush(self):
        """Send the accumulated transcript fragments as one message."""
        self._flush_handle = None
        if not self._buf:
            return
        text = " ".join(self._buf)
        self._buf.clear()
        await _send_transcript(self._transport, "user", text)


class AssistantTranscriptForwarder(FrameProcessor):
    """Place after LLM to capture the full assistant response.
    Buffers streaming TextFrames between LLMFullResponseStart/End."""

    def __init__(self, transport: LiveKitTransport, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport
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
                await _send_transcript(self._transport, "assistant", full)
            self._buf.clear()

        await self.push_frame(frame, direction)

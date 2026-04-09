"""Frame processors that forward transcript data to the web UI via LiveKit data channel."""

from __future__ import annotations

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


async def _send_transcript(transport: LiveKitTransport, role: str, text: str):
    msg = json.dumps({"role": role, "text": text}, ensure_ascii=False)
    logger.debug("Transcript → client: {msg}", msg=msg)
    await transport.send_message(msg)


class UserTranscriptForwarder(FrameProcessor):
    """Place between STT and user_agg to capture user transcriptions
    before user_agg consumes them."""

    def __init__(self, transport: LiveKitTransport, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await _send_transcript(self._transport, "user", frame.text.strip())
        await self.push_frame(frame, direction)


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

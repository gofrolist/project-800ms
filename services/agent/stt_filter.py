"""Filter out common Whisper hallucinations on silence/noise.

Whisper is notorious for producing phantom transcriptions like "Thank you.",
"Thanks for watching.", "Bye." when fed silence or low-level background noise.
This processor drops those before they reach the LLM.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Lowercase, stripped. Whisper produces these on silence across many languages.
HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thank you",
        "thanks.",
        "thanks",
        "thanks for watching.",
        "thanks for watching",
        "bye.",
        "bye",
        "goodbye.",
        "goodbye",
        "you",
        "the end.",
        "the end",
        "so",
        "okay.",
        "okay",
        "oh",
        "ah",
        "hmm",
        "uh",
        "um",
        # Russian equivalents
        "спасибо.",
        "спасибо",
        "пока.",
        "пока",
        "до свидания.",
        "до свидания",
        "продолжение следует...",
        "продолжение следует",
        "продолжение следует…",
        "субтитры сделал dyadefima",
        "субтитры добавил dyadefima",
    }
)


class WhisperHallucinationFilter(FrameProcessor):
    """Drops TranscriptionFrames that match known Whisper hallucination phrases."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            normalized = frame.text.strip().lower()
            if normalized in HALLUCINATIONS:
                logger.debug("Dropped Whisper hallucination: {text}", text=frame.text)
                return  # swallow the frame
        await self.push_frame(frame, direction)

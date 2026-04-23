"""Small shared helpers for in-process TTS adapters.

The heavy adapters (``silero_tts.py``, ``xtts_tts.py``) previously kept
their own byte-for-byte copies of these utilities. Extracting them here
gives a single source of truth so a future change to Pipecat's framing
contract needs exactly one edit.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator


async def single_shot_audio_iterator(pcm: bytes) -> AsyncGenerator[bytes, None]:
    """Wrap a complete PCM buffer as a one-yield async generator.

    ``TTSService._stream_audio_frames_from_iterator`` expects an
    async-iter of byte chunks — file-at-a-time synths (Silero, XTTS)
    render the whole utterance off-thread into a single buffer, then
    hand it back through this helper so the framing plumbing can
    chunk it into ``TTSAudioRawFrame`` at the service's sample rate.
    """
    yield pcm

"""POST final transcripts to the API's /internal/transcripts endpoint.

Kept in its own module (no pipecat imports) so it can be unit-tested
without CUDA. A best-effort sink — a failed POST is logged but never
propagates into the Pipecat pipeline, so a flaky API never kills a
live voice session.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import aiohttp
from loguru import logger


@dataclass(frozen=True)
class TranscriptSink:
    """Logs one transcript per utterance to the API.

    Configure with the room name and auth context; call `log` from
    whichever frame processor finalizes an utterance.
    """

    api_base_url: str  # e.g. "http://api:8000"
    internal_token: str
    room: str
    timeout_seconds: float = 2.0

    async def log(self, role: str, text: str) -> None:
        """Post one utterance. Swallows exceptions — never let a transcript
        write fail a live voice session."""
        if not self.api_base_url or not self.internal_token:
            # Misconfigured — caller asked for a sink but didn't supply
            # creds. Log once and degrade silently so the pipeline runs.
            return

        url = f"{self.api_base_url.rstrip('/')}/internal/transcripts"
        payload = {
            "room": self.room,
            "role": role,
            "text": text,
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"X-Internal-Token": self.internal_token},
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.warning(
                            "Transcript sink non-2xx status={status} body={body}",
                            status=resp.status,
                            body=body[:200],
                        )
        except Exception:
            # Network hiccup, timeout, DNS glitch — don't propagate.
            logger.exception("Transcript sink POST failed (room={room})", room=self.room)

# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Bearer-token authentication dependency for the OpenAI-compatible router.

Local patch on top of upstream Qwen3-TTS-Openai-Fastapi
(eb14f6e6a50445cf442979abb9203ff0d5042c43). Upstream does not
authenticate the /v1/audio/speech endpoint — in our docker-compose
deploy the sidecar is bound to 127.0.0.1 but still reachable from
every other container on the internal network, so we enforce a shared
secret.

Backwards-compatible: when TTS_API_KEY is unset, the dependency is a
no-op and all requests pass through (same behavior as upstream). To
enable auth, the compose environment sets TTS_API_KEY on the sidecar
and QWEN3_TTS_API_KEY on the agent; Pipecat's OpenAI client sends the
same value via the Authorization header.
"""

import os

from fastapi import Header, HTTPException, status

TTS_API_KEY = os.getenv("TTS_API_KEY", "")
"""
Shared bearer token the agent sends in the Authorization header.
Empty string disables auth (upstream-compatible fallback); any
non-empty value enables strict check.
"""


async def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency. Validate a bearer token against TTS_API_KEY.

    Raises 401 when:
    - TTS_API_KEY is set and Authorization header is missing or not Bearer
    - the bearer value does not exactly match TTS_API_KEY

    Returns None (no-op) when TTS_API_KEY is empty — upstream behavior.
    """
    if not TTS_API_KEY:
        # Key not configured on the sidecar — allow all requests.
        # Operator must set TTS_API_KEY in the compose environment to
        # enable the auth check.
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Missing or malformed Authorization header",
                "type": "invalid_request_error",
            },
        )

    token = authorization[len("Bearer "):].strip()
    if token != TTS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Invalid API key",
                "type": "invalid_request_error",
            },
        )

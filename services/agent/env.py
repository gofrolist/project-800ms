"""Tiny env-var helper. Lives in its own module so it can be unit-tested
without pulling in pipecat / gigaam / CUDA at import time."""

from __future__ import annotations

import os


class MissingEnvError(RuntimeError):
    """Raised when a required environment variable is missing or empty."""


def require_env(name: str, default: str | None = None) -> str:
    """Return the value of env var `name`, or `default` if it is unset.

    Raises ``MissingEnvError`` if both the env var and the default are
    missing. Empty strings count as missing — an empty credential is almost
    always a misconfiguration, not an intentional blank.
    """
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise MissingEnvError(f"Missing required env var: {name}")
    return value

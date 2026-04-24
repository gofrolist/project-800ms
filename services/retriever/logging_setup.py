"""Structured logging for the retriever.

Uses loguru with lazy `{name}` placeholders (NOT f-strings) on the hot
path — constitution Principle V. Per-turn contextvars (tenant_id /
session_id / turn_id) are set via `request_context()` (a contextmanager)
so the pair is always balanced: enter sets via `ContextVar.set()` and
captures the returned tokens; exit always calls `Token.reset()`, even on
exception. This eliminates the bind/clear-leak class of bug.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from loguru import logger

_TENANT_ID: ContextVar[str] = ContextVar("tenant_id", default="")
_SESSION_ID: ContextVar[str] = ContextVar("session_id", default="")
_TURN_ID: ContextVar[str] = ContextVar("turn_id", default="")


def configure_logging(level: str = "INFO") -> None:
    """Replace the default loguru sink with a JSON-ish line format that
    includes the per-request context.

    Idempotency is owned by the caller (`main.py` guards repeat calls);
    this function unconditionally reconfigures loguru.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "tenant={extra[tenant_id]} session={extra[session_id]} "
            "turn={extra[turn_id]} | {message}"
        ),
        enqueue=False,
        backtrace=False,
        diagnose=False,  # diagnose=True would print local variables — unsafe.
    )
    logger.configure(patcher=_context_patcher)


def _context_patcher(record: dict) -> None:  # type: ignore[type-arg]
    """Inject contextvars into every log record's `extra` dict."""
    extra = record.get("extra", {})
    extra.setdefault("tenant_id", _TENANT_ID.get())
    extra.setdefault("session_id", _SESSION_ID.get())
    extra.setdefault("turn_id", _TURN_ID.get())
    record["extra"] = extra


@contextmanager
def request_context(
    *,
    tenant_id: str = "",
    session_id: str = "",
    turn_id: str = "",
) -> Iterator[None]:
    """Bind per-request context for the enclosed block.

    Always balanced: `ContextVar.set()` tokens are captured and
    `Token.reset()` runs in `finally`, even on exception. Use from any
    route handler:

        async with request_context(tenant_id=..., session_id=..., turn_id=...):
            # handler body logs carry the bound context.

    Note: the wrapper is a sync contextmanager but safe to use with
    `async with` via `contextlib.AsyncExitStack` if needed; for a simple
    FastAPI handler, the sync form works inside `async def` via `with`.
    """
    bindings: list[tuple[ContextVar[str], Token[str]]] = [
        (_TENANT_ID, _TENANT_ID.set(tenant_id)),
        (_SESSION_ID, _SESSION_ID.set(session_id)),
        (_TURN_ID, _TURN_ID.set(turn_id)),
    ]
    try:
        yield
    finally:
        # Reset in reverse order of acquisition so nested contexts pop cleanly.
        for var, token in reversed(bindings):
            var.reset(token)

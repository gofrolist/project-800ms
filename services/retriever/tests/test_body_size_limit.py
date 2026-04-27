"""Body-size middleware tests (issue #56).

Asserts the cap fires BEFORE the auth dep so an unauthenticated
caller can't OOM the retriever via Content-Length probes. Returns
the flat error envelope shape (`{error, message}`).

Fast tests — no DB, no LLM, just a minimal FastAPI app + ASGI
transport.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport


@pytest_asyncio.fixture
async def app(monkeypatch):
    """Minimal retriever app — env baseline already set by the
    session-autouse fixture in conftest.py."""
    monkeypatch.setenv("RETRIEVER_INTERNAL_TOKEN", "test-internal-token")

    import config
    import db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db._session_factory.cache_clear()

    import sys

    sys.modules.pop("main", None)
    sys.modules.pop("retrieve", None)
    from main import create_app

    return create_app()


async def test_oversized_body_returns_413_before_auth(app) -> None:
    """A 65-KiB body is rejected with 413 even WITHOUT a token —
    the middleware fires before auth, so the auth dep never runs.
    Closes the pre-auth DoS vector documented in issue #56.
    """
    transport = ASGITransport(app=app)
    big_body = "x" * (65 * 1024)
    async with httpx.AsyncClient(transport=transport, base_url="http://retriever.test") as client:
        resp = await client.post(
            "/retrieve",
            content=big_body,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert body["error"]["code"] == "payload_too_large"
    # Critical: the response is NOT 401 — the auth dep didn't even run.
    # If a future refactor moves the body-size check after auth, this
    # assertion would flip to 401 and we'd notice.


async def test_normal_body_size_passes_through(app) -> None:
    """A typical request size sails through the middleware. The auth
    dep then rejects it with 401 since we sent no token — confirming
    the middleware is non-disruptive on legitimate traffic shapes.
    """
    transport = ASGITransport(app=app)
    normal_body = '{"tenant_id":"00000000-0000-0000-0000-000000000001","session_id":"00000000-0000-0000-0000-000000000002","turn_id":"t-1","transcript":"hi"}'
    async with httpx.AsyncClient(transport=transport, base_url="http://retriever.test") as client:
        resp = await client.post(
            "/retrieve",
            content=normal_body,
            headers={"content-type": "application/json"},
        )
    # Auth dep wins — body wasn't rejected by size.
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "unauthenticated"


@pytest.mark.parametrize("invalid_length", ["not-a-number", "-5", ""])
async def test_invalid_content_length_passes_through(app, invalid_length) -> None:
    """Non-numeric or negative Content-Length passes the middleware
    (treated as length 0); subsequent layers handle it. Required so
    a buggy / hostile header value can't break legitimate traffic
    paths — only an explicit oversize triggers the cap.
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://retriever.test") as client:
        resp = await client.post(
            "/retrieve",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": invalid_length,
            },
        )
    # Either 401 (auth wins) or 400 (Pydantic) — anything but 413.
    assert resp.status_code != 413, resp.text

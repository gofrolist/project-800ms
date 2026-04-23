"""GET /v1/engines — tests for the engine-discovery endpoint.

Uses the real FastAPI app with ``httpx.AsyncClient`` patched so the
agent /engines response is deterministic. Mirrors the
``test_v1_sessions.py`` fixture shape for consistency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Request, Response

from auth import _clear_cache_for_tests
from main import app

pytestmark = pytest.mark.slow


@pytest.fixture
def client(override_db):
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


@pytest.fixture
def override_db(db_session):
    from db import get_db

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    _clear_cache_for_tests()
    yield
    app.dependency_overrides.clear()


def _agent_response(payload, status_code=200):
    """Build an httpx ``Response`` with the given JSON body."""
    return Response(
        status_code,
        json=payload,
        request=Request("GET", "http://agent/engines"),
    )


@pytest.fixture
def agent_engines_stub(monkeypatch):
    """Patch ``routes.engines.httpx.AsyncClient`` so GET /engines on
    the agent returns a deterministic response for each test. Default
    payload matches the full four-engine demo deploy.
    """
    with patch("routes.engines.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(
            return_value=_agent_response(
                {
                    "available": ["piper", "silero", "qwen3", "xtts"],
                    "default": "piper",
                }
            )
        )
        mock_client_class.return_value = mock_client
        yield mock_client


async def test_engines_requires_auth(client, agent_engines_stub):
    """Same auth contract as the rest of /v1/* — missing key → 401."""
    r = await client.get("/v1/engines")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_engines_returns_full_list(client, seed_tenant, agent_engines_stub):
    """Happy path: agent reports all four engines available, API
    returns each with the static metadata layered on top."""
    _tenant, raw_key = seed_tenant
    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "engines" in body

    engines_by_id = {e["id"]: e for e in body["engines"]}
    assert set(engines_by_id.keys()) == {"piper", "silero", "qwen3", "xtts"}
    # All available when the agent advertises them all.
    assert all(e["available"] for e in body["engines"])
    # piper is the default in the stub payload.
    assert engines_by_id["piper"]["default"] is True
    assert engines_by_id["silero"]["default"] is False

    # Static per-engine metadata is layered correctly.
    assert engines_by_id["piper"]["voice_format"] == "piper_voice_name"
    assert engines_by_id["silero"]["voice_format"] == "silero_speaker_id"
    assert engines_by_id["qwen3"]["voice_format"] == "openai_or_clone"
    assert engines_by_id["xtts"]["voice_format"] == "clone_only"


async def test_engines_reflects_partial_availability(client, seed_tenant, agent_engines_stub):
    """When the agent's preload set is partial (e.g. XTTS degraded out
    after low-disk check), the API response marks the missing engines
    as available=False instead of omitting them. Clients can still see
    xtts exists as a type but know it's not bookable on this deploy.
    """
    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.return_value = _agent_response(
        {"available": ["piper", "silero"], "default": "piper"}
    )

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 200
    engines_by_id = {e["id"]: e for e in r.json()["engines"]}

    # Every known engine appears in the list, with availability
    # matching the agent's preload set.
    assert engines_by_id["piper"]["available"] is True
    assert engines_by_id["silero"]["available"] is True
    assert engines_by_id["qwen3"]["available"] is False
    assert engines_by_id["xtts"]["available"] is False


async def test_engines_default_flag_tracks_agent_tts_engine(
    client, seed_tenant, agent_engines_stub
):
    """``default: true`` attaches to whichever engine matches the
    agent's TTS_ENGINE env. Exactly one engine (or zero) has the flag."""
    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.return_value = _agent_response(
        {"available": ["xtts"], "default": "xtts"}
    )

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    body = r.json()
    defaults = [e["id"] for e in body["engines"] if e["default"]]
    assert defaults == ["xtts"]


async def test_engines_503_when_agent_unreachable(client, seed_tenant, agent_engines_stub):
    """Agent down → 503 with the canonical error envelope, not 500.
    Clients can retry."""
    import httpx  # noqa: PLC0415

    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.side_effect = httpx.ConnectError("connection refused")

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "agent_unavailable"


async def test_engines_503_when_agent_returns_non_200(client, seed_tenant, agent_engines_stub):
    """Agent returns non-200 (e.g. 500 from a bug in handle_engines) →
    503 rather than cargo-culting the agent's status through. Keeps the
    public API's status taxonomy clean."""
    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.return_value = _agent_response({"error": "internal"}, status_code=500)

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 503


async def test_engines_503_when_agent_returns_invalid_json(client, seed_tenant, agent_engines_stub):
    """Malformed agent response → 503, not 500."""
    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.return_value = Response(
        200,
        text="not json",
        request=Request("GET", "http://agent/engines"),
    )

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 503


async def test_engines_handles_agent_missing_fields_gracefully(
    client, seed_tenant, agent_engines_stub
):
    """If the agent response is malformed (missing available/default),
    the API returns 200 with everything marked unavailable rather than
    500ing. Clients can then show 'no engines ready' to the user.
    """
    _tenant, raw_key = seed_tenant
    agent_engines_stub.get.return_value = _agent_response({})

    r = await client.get("/v1/engines", headers={"X-API-Key": raw_key})
    assert r.status_code == 200
    body = r.json()
    assert all(e["available"] is False for e in body["engines"])
    assert all(e["default"] is False for e in body["engines"])

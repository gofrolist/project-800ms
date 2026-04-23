"""POST /v1/sessions + GET /v1/sessions/{room} tests.

Uses the real app (so middleware, error envelope, auth all match prod)
with two overrides:
    - get_db is redirected to the savepointed db_session fixture
    - httpx.AsyncClient.post is patched to stub the agent dispatch call

Requires the Postgres testcontainer — marked slow.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Request, Response
from sqlalchemy import select

from auth import _clear_cache_for_tests
from db import get_db
from main import app
from models import Session as SessionRow

pytestmark = pytest.mark.slow


@pytest.fixture
def override_db(db_session):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    _clear_cache_for_tests()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_db):
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


@pytest.fixture
def agent_stub():
    """Patch httpx.AsyncClient.post so /dispatch always succeeds."""
    with patch("routes.sessions.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(
            return_value=Response(
                200,
                json={"status": "dispatched"},
                request=Request("POST", "http://agent/dispatch"),
            )
        )
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def agent_down():
    """Patch httpx.AsyncClient.post to raise HTTP errors (simulate agent down)."""
    import httpx

    with patch("routes.sessions.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client_class.return_value = mock_client
        yield mock_client


async def test_create_session_requires_auth(client):
    r = await client.post("/v1/sessions", json={})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_create_session_empty_body_works(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body.keys()) == {"session_id", "room", "identity", "token", "url"}
    assert body["room"].startswith("room-")
    assert body["identity"].startswith("user-")
    assert body["token"]  # JWT is non-empty
    assert agent_stub.post.await_count == 1


async def test_create_session_persists_row(client, db_session, seed_tenant, agent_stub):
    tenant, raw_key = seed_tenant
    payload: dict[str, Any] = {
        "user_id": "player-42",
        "npc_id": "merchant",
        "persona": {"backstory": "grumpy shopkeeper"},
        "voice": "ru_RU-denis-medium",
        "language": "ru",
        "llm_model": "llama-3.3-70b-versatile",
        "context": {"inventory": ["sword", "shield"]},
    }
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json=payload)
    assert r.status_code == 201, r.json()

    row = (
        await db_session.execute(select(SessionRow).where(SessionRow.room == r.json()["room"]))
    ).scalar_one()
    assert row.user_id == "player-42"
    assert row.npc_id == "merchant"
    assert row.persona == {"backstory": "grumpy shopkeeper"}
    assert row.voice == "ru_RU-denis-medium"
    assert row.language == "ru"
    assert row.llm_model == "llama-3.3-70b-versatile"
    assert row.context == {"inventory": ["sword", "shield"]}
    assert row.status == "active"
    assert row.started_at is not None
    assert str(row.tenant_id) == str(tenant.id)
    # user_id becomes the caller identity when provided.
    assert row.identity == "player-42"


async def test_create_session_forwards_payload_to_agent(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    payload = {"npc_id": "merchant", "voice": "ru_RU-denis-medium"}
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json=payload)
    assert r.status_code == 201

    sent = agent_stub.post.await_args
    dispatched = sent.kwargs["json"]
    assert dispatched["npc_id"] == "merchant"
    assert dispatched["voice"] == "ru_RU-denis-medium"
    assert dispatched["room"].startswith("room-")
    # Fields not provided shouldn't appear at all — keeps the agent-side
    # contract tight.
    assert "persona" not in dispatched
    assert "context" not in dispatched


async def test_create_session_forwards_tts_engine_to_agent(client, seed_tenant, agent_stub):
    """Three-button demo contract: POST /v1/sessions with tts_engine flows
    through the dispatch payload so the agent can pick the right TTS
    backend per-session."""
    _tenant, raw_key = seed_tenant
    for engine in ("piper", "silero", "qwen3", "xtts"):
        agent_stub.post.reset_mock()
        r = await client.post(
            "/v1/sessions",
            headers={"X-API-Key": raw_key},
            json={"tts_engine": engine},
        )
        assert r.status_code == 201, r.text
        dispatched = agent_stub.post.await_args.kwargs["json"]
        assert dispatched["tts_engine"] == engine


async def test_create_session_omits_tts_engine_when_unset(client, seed_tenant, agent_stub):
    """Empty body must NOT inject tts_engine — the agent's TTS_ENGINE env
    default has to win when the client is silent about engine choice."""
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 201
    dispatched = agent_stub.post.await_args.kwargs["json"]
    assert "tts_engine" not in dispatched


async def test_create_session_rejects_unknown_tts_engine(client, seed_tenant, agent_stub):
    """Pydantic Literal validation refuses unknown engine values BEFORE
    the dispatch call — keeps bogus strings from ever reaching the agent."""
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"tts_engine": "bogus"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert agent_stub.post.await_count == 0


async def test_create_session_rejects_xtts_without_clone_voice(client, seed_tenant, agent_stub):
    """XTTS v2 is voice-cloning only — a non-clone: voice with
    tts_engine=xtts must be rejected as a 422 at the API rather than
    producing a silent dead session.

    Without this check: API returns 201 + LiveKit token, the caller
    joins the room, the agent's _resolve_voice_profile raises
    ValueError inside _run_pipeline, that exception is swallowed by
    the bare except, and the user hears silence with no HTTP error.
    """
    _tenant, raw_key = seed_tenant

    # "alloy" is a legitimate Qwen3 voice name — ensure the cross-engine
    # collision is caught. A Silero speaker id like "ru_dmitriy" should
    # also fail.
    for bad_voice in ("alloy", "ru_dmitriy", "ru_RU-denis-medium"):
        agent_stub.post.reset_mock()
        r = await client.post(
            "/v1/sessions",
            headers={"X-API-Key": raw_key},
            json={"tts_engine": "xtts", "voice": bad_voice},
        )
        assert r.status_code == 422, (
            f"expected 422 for voice={bad_voice!r}, got {r.status_code}; body={r.text[:400]!r}"
        )
        assert r.json()["error"]["code"] == "validation_error"
        # Must not reach the agent.
        assert agent_stub.post.await_count == 0


async def test_create_session_accepts_xtts_with_clone_voice(client, seed_tenant, agent_stub):
    """Baseline: tts_engine=xtts + voice=clone:<profile> must pass
    validation and reach the agent. Pins that the validator doesn't
    over-reject.
    """
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"tts_engine": "xtts", "voice": "clone:demo-ru"},
    )
    assert r.status_code == 201, r.text
    dispatched = agent_stub.post.await_args.kwargs["json"]
    assert dispatched["tts_engine"] == "xtts"
    assert dispatched["voice"] == "clone:demo-ru"


async def test_create_session_xtts_without_voice_reaches_agent(client, seed_tenant, agent_stub):
    """Omitting voice entirely is legal at the API — the agent falls
    back to XTTS_TTS_VOICE / TTS_VOICE env defaults. The validator
    only rejects EXPLICIT non-clone values with xtts."""
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"tts_engine": "xtts"},
    )
    assert r.status_code == 201, r.text
    dispatched = agent_stub.post.await_args.kwargs["json"]
    assert dispatched["tts_engine"] == "xtts"
    assert "voice" not in dispatched


async def test_create_session_allows_alloy_with_qwen3(client, seed_tenant, agent_stub):
    """Regression guard: the xtts+clone-only constraint must not leak
    into the qwen3 engine path. ``voice="alloy"`` + ``tts_engine="qwen3"``
    is the canonical Qwen3 CustomVoice request and must keep working.
    """
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"tts_engine": "qwen3", "voice": "alloy"},
    )
    assert r.status_code == 201, r.text
    dispatched = agent_stub.post.await_args.kwargs["json"]
    assert dispatched["tts_engine"] == "qwen3"
    assert dispatched["voice"] == "alloy"


async def test_create_session_rejects_unknown_fields(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"bogus_field": "value"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    # extra="forbid" must refuse the unknown field before we hit the agent.
    assert agent_stub.post.await_count == 0


async def test_create_session_agent_down_returns_503_and_rolls_back(
    client, db_session, seed_tenant, agent_down
):
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "agent_unavailable"

    # No session row should have been persisted — the insert was rolled back.
    rows = (await db_session.execute(select(SessionRow))).scalars().all()
    assert rows == []


async def test_get_session_returns_details(client, seed_tenant, agent_stub):
    _tenant, raw_key = seed_tenant
    r = await client.post(
        "/v1/sessions",
        headers={"X-API-Key": raw_key},
        json={"npc_id": "merchant"},
    )
    room = r.json()["room"]

    g = await client.get(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert g.status_code == 200
    body = g.json()
    assert body["room"] == room
    assert body["npc_id"] == "merchant"
    assert body["status"] == "active"


async def test_get_session_cross_tenant_returns_404(client, db_session, seed_tenant, agent_stub):
    """Tenant A cannot fetch tenant B's session — we return 404, not 403,
    to avoid leaking existence."""
    import hashlib
    import uuid as _uuid

    from models import ApiKey, Tenant

    _tenant_a, key_a = seed_tenant

    # Spin up tenant B with its own key.
    raw_b = "tk_b_" + _uuid.uuid4().hex
    tenant_b = Tenant(name="B", slug=f"b-{_uuid.uuid4().hex[:8]}")
    db_session.add(tenant_b)
    await db_session.flush()
    db_session.add(
        ApiKey(
            tenant_id=tenant_b.id,
            key_hash=hashlib.sha256(raw_b.encode()).digest(),
            key_prefix=raw_b[:8],
        )
    )
    await db_session.flush()

    # Tenant A creates a session.
    r = await client.post("/v1/sessions", headers={"X-API-Key": key_a}, json={})
    room = r.json()["room"]

    # Tenant B cannot see it.
    g = await client.get(f"/v1/sessions/{room}", headers={"X-API-Key": raw_b})
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "not_found"


async def test_get_session_not_found(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    g = await client.get("/v1/sessions/room-doesnotexist", headers={"X-API-Key": raw_key})
    assert g.status_code == 404
    assert g.json()["error"]["code"] == "not_found"


async def test_legacy_sessions_endpoint_removed(client):
    """Burn-the-boats: POST /sessions no longer exists.

    Starlette's router produces a plain {"detail": "Not Found"} for
    unmatched paths — that's fine. The envelope is guaranteed only for
    /v1/* responses that pass through our handlers.
    """
    r = await client.post("/sessions", json={})
    assert r.status_code == 404


# ─── DELETE /v1/sessions/{room} ─────────────────────────────────────────────
# Stubs the LiveKit RoomService.delete_room call so tests don't reach a real
# LiveKit server. The route's own _delete_livekit_room helper constructs a
# ``lkapi.LiveKitAPI(...)`` inside ``async with``, so patching the class
# itself (and not the individual method) is the cleanest interception point.


@pytest.fixture
def livekit_stub():
    """Patch lkapi.LiveKitAPI so delete_room appears to succeed."""
    with patch("routes.sessions.lkapi.LiveKitAPI") as mock_lk_class:
        mock_lk = AsyncMock()
        mock_lk.__aenter__.return_value = mock_lk
        mock_lk.__aexit__.return_value = None
        mock_lk.room = AsyncMock()
        mock_lk.room.delete_room = AsyncMock(return_value=None)
        mock_lk_class.return_value = mock_lk
        yield mock_lk


@pytest.fixture(
    params=[
        pytest.param("connect_error", id="aiohttp-connect-error"),
        pytest.param("timeout", id="asyncio-timeout"),
        pytest.param("os_error", id="os-error"),
    ]
)
def livekit_down(request):
    """Patch lkapi.LiveKitAPI to simulate a LiveKit outage on delete_room.

    Parameterized over the specific exception families ``_delete_livekit_room``
    narrowly catches — aiohttp client error, asyncio timeout, and raw
    OSError — so a regression that drops one of these from the except
    tuple (or lets an unrelated error class slip in that should
    propagate) shows up as a concrete test failure rather than
    'somehow DELETE still worked anyway'.
    """
    import aiohttp  # noqa: PLC0415
    import asyncio  # noqa: PLC0415

    exceptions = {
        "connect_error": aiohttp.ClientConnectionError("cannot connect"),
        "timeout": asyncio.TimeoutError(),
        "os_error": OSError("network unreachable"),
    }
    with patch("routes.sessions.lkapi.LiveKitAPI") as mock_lk_class:
        mock_lk = AsyncMock()
        mock_lk.__aenter__.return_value = mock_lk
        mock_lk.__aexit__.return_value = None
        mock_lk.room = AsyncMock()
        mock_lk.room.delete_room = AsyncMock(side_effect=exceptions[request.param])
        mock_lk_class.return_value = mock_lk
        yield mock_lk


async def test_delete_session_requires_auth(client):
    r = await client.delete("/v1/sessions/room-whatever")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_delete_session_not_found(client, seed_tenant):
    _tenant, raw_key = seed_tenant
    r = await client.delete(
        "/v1/sessions/room-doesnotexist",
        headers={"X-API-Key": raw_key},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_delete_session_cross_tenant_returns_404(
    client, db_session, seed_tenant, agent_stub, livekit_stub
):
    """Tenant A cannot delete tenant B's session — 404 (same opacity
    rule as GET). Tenant B's room must remain active in LiveKit, so the
    LiveKit stub should never see a delete_room call."""
    import hashlib
    import uuid as _uuid

    from models import ApiKey, Tenant

    _tenant_a, key_a = seed_tenant

    raw_b = "tk_b_" + _uuid.uuid4().hex
    tenant_b = Tenant(name="B", slug=f"b-{_uuid.uuid4().hex[:8]}")
    db_session.add(tenant_b)
    await db_session.flush()
    db_session.add(
        ApiKey(
            tenant_id=tenant_b.id,
            key_hash=hashlib.sha256(raw_b.encode()).digest(),
            key_prefix=raw_b[:8],
        )
    )
    await db_session.flush()

    r = await client.post("/v1/sessions", headers={"X-API-Key": key_a}, json={})
    room = r.json()["room"]

    # Tenant B tries to delete — forbidden, returns 404 to avoid leaking existence.
    d = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_b})
    assert d.status_code == 404
    assert d.json()["error"]["code"] == "not_found"
    # LiveKit must NOT have been touched for a cross-tenant attempt.
    livekit_stub.room.delete_room.assert_not_called()


async def test_delete_session_ends_active_session(
    client, db_session, seed_tenant, agent_stub, livekit_stub
):
    """Happy path: DELETE on an active session → LiveKit delete_room is
    called, DB row flips to status='ended' with ended_at AND audio_seconds
    set, response body reflects the new state.

    The audio_seconds assertion is load-bearing: before this field was
    computed on DELETE, the room_finished webhook's
    ``if session.ended_at is None`` short-circuit skipped the calculation
    for every user-initiated hang-up, silently producing NULL
    audio_seconds that usage.py coalesces to 0 — every UI close
    under-billed. This test pins the fix.
    """
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    room = r.json()["room"]

    d = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d.status_code == 200, d.text
    body = d.json()
    assert body["room"] == room
    assert body["status"] == "ended"
    assert body["ended_at"] is not None
    # audio_seconds is computed from started_at → ended_at. The test
    # post→delete path takes milliseconds, so the value is 0 or 1 —
    # what we need to pin is that it's populated (not None), not the
    # exact magnitude.
    assert body["audio_seconds"] is not None
    assert body["audio_seconds"] >= 0

    # DB row is consistent with the response.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    assert row.status == "ended"
    assert row.ended_at is not None
    assert row.audio_seconds is not None
    assert row.audio_seconds >= 0

    # LiveKit was told to tear down the room.
    livekit_stub.room.delete_room.assert_called_once()
    delete_request = livekit_stub.room.delete_room.call_args.args[0]
    assert delete_request.room == room


async def test_delete_session_is_idempotent(
    client, db_session, seed_tenant, agent_stub, livekit_stub
):
    """Second DELETE on the same room must return 200 with the existing
    ended state, NOT re-call LiveKit (the room is gone) and NOT re-write
    ended_at (preserves the original close timestamp for billing/audit)."""
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    room = r.json()["room"]

    d1 = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d1.status_code == 200
    first_ended_at = d1.json()["ended_at"]
    assert first_ended_at is not None
    assert livekit_stub.room.delete_room.call_count == 1

    # Second delete — should be a no-op.
    d2 = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d2.status_code == 200
    body = d2.json()
    assert body["status"] == "ended"
    # ended_at MUST be stable across retries — changing it would break
    # any usage/billing report that joins on that timestamp.
    assert body["ended_at"] == first_ended_at
    # No second LiveKit tear-down call — the room is already gone.
    assert livekit_stub.room.delete_room.call_count == 1


async def test_delete_session_survives_livekit_outage(
    client, db_session, seed_tenant, agent_stub, livekit_down
):
    """If the LiveKit API is unavailable, DELETE must still mark the
    session ``ended`` in the DB rather than returning 503. Rationale
    (see _delete_livekit_room docstring): the primary teardown path is
    the agent's own on_participant_left handler, and we don't want a
    transient LiveKit outage to block the caller's "hang up" — the room
    will fall out on LiveKit's own empty_timeout once the browser
    disconnects."""
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    room = r.json()["room"]

    d = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d.status_code == 200, d.text
    assert d.json()["status"] == "ended"

    # DB is still consistent even though LiveKit raised.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    assert row.status == "ended"
    assert row.ended_at is not None


async def test_delete_session_on_pending_returns_409(
    client, db_session, seed_tenant, agent_stub, livekit_stub
):
    """DELETE must refuse to close a session still in ``pending``.

    A pending row has ``started_at=NULL``; usage.py filters on
    ``started_at >= start`` so marking it ended would make the call
    invisible to usage reports. We return 409 and tell the client to
    retry once POST commits. Under normal concurrency this window is
    tiny (pending → active transitions synchronously inside
    create_session before 201), but the route must defend the invariant
    rather than assume the window never manifests.
    """
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    room = r.json()["room"]

    # Flip the row back to 'pending' + clear started_at to simulate the
    # narrow window before create_session's pending → active transition
    # commits.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    row.status = "pending"
    row.started_at = None
    await db_session.commit()

    d = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d.status_code == 409
    assert d.json()["error"]["code"] == "conflict"
    # LiveKit must NOT have been called — we refused the close outright.
    livekit_stub.room.delete_room.assert_not_called()
    # Row must still be pending — we didn't leak a partial write.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    assert row.status == "pending"
    assert row.ended_at is None


async def test_delete_session_preserves_failed_status(
    client, db_session, seed_tenant, agent_stub, livekit_stub
):
    """DELETE on a session already in ``failed`` must NOT overwrite it
    with ``ended`` — ``failed`` is a terminal diagnostic state for
    incident triage (e.g., pipeline crashed, agent dispatch errored).
    Letting DELETE stomp it would erase the signal on every retry.
    """
    _tenant, raw_key = seed_tenant
    r = await client.post("/v1/sessions", headers={"X-API-Key": raw_key}, json={})
    room = r.json()["room"]

    # Flip the row to 'failed' directly — simulates the pipeline having
    # crashed before the operator runs DELETE.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    row.status = "failed"
    row.ended_at = datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=datetime.UTC)
    await db_session.commit()

    d = await client.delete(f"/v1/sessions/{room}", headers={"X-API-Key": raw_key})
    assert d.status_code == 200
    body = d.json()
    assert body["status"] == "failed"  # preserved, not flipped to 'ended'
    assert body["ended_at"] is not None

    # LiveKit NOT called — we short-circuited on the terminal status.
    livekit_stub.room.delete_room.assert_not_called()

    # DB row still shows 'failed'.
    row = (await db_session.execute(select(SessionRow).where(SessionRow.room == room))).scalar_one()
    assert row.status == "failed"

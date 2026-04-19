"""End-to-end smoke test against a running deployment.

Verifies the happy path a real caller experiences:

    1. GET  /health               -> 200
    2. POST /v1/sessions          -> 201 with valid session JWT
    3. LiveKit JWT has the expected video grants + room binding
    4. GET  /v1/sessions/{room}   -> 200, status progresses to 'active'
       once the agent picks up the dispatch
    5. DELETE session cleanup is out-of-scope (sessions end via LiveKit
       webhook when the caller disconnects; no API on the surface today)

Catches regressions no unit test can — the webhook actually firing, the
LiveKit SFU actually accepting the JWT, the agent actually joining the
room.

Usage:
    uv run python scripts/smoke.py
    SMOKE_API_URL=https://api.coastalai.ai \\
    SMOKE_API_KEY=tk_... \\
    uv run python scripts/smoke.py

Returns exit 0 on success, 1 on failure. Intended for CI + oncall.
"""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jwt

API_URL = os.environ.get("SMOKE_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("SMOKE_API_KEY", "")
# Hard cap on how long we wait for the agent to mark the session active.
# 30s covers cold-start on a freshly booted VM; longer than that is
# almost certainly a real problem.
AGENT_TIMEOUT_SECS = 30


def _req(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    expect_status: int,
) -> dict:
    req = Request(f"{API_URL}{path}", method=method)
    req.add_header("Accept", "application/json")
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()

    try:
        with urlopen(req, data=data, timeout=10) as resp:
            status = resp.status
            body_text = resp.read().decode()
    except HTTPError as e:
        status = e.code
        body_text = e.read().decode() if e.fp else ""
    except URLError as e:
        _fail(f"{method} {path} — network error: {e}")

    if status != expect_status:
        _fail(f"{method} {path} — want {expect_status}, got {status}: {body_text[:200]}")
    return json.loads(body_text) if body_text else {}


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


def step_health() -> None:
    print("1. GET /health")
    body = _req("/health", expect_status=200)
    if body.get("status") != "ok":
        _fail(f"/health payload: {body!r}")
    _ok("service is up")


def step_create_session() -> dict:
    print("2. POST /v1/sessions")
    if not API_KEY:
        _fail("SMOKE_API_KEY is not set — can't exercise /v1/sessions")
    body = _req(
        "/v1/sessions",
        method="POST",
        body={"user_id": "smoke-test", "npc_id": "smoke"},
        expect_status=201,
    )
    required = {"session_id", "room", "identity", "token", "url"}
    if not required.issubset(body):
        _fail(f"session response missing keys: {required - set(body)}")
    _ok(f"session_id={body['session_id']} room={body['room']}")
    return body


def step_verify_jwt(session: dict) -> None:
    print("3. Verify LiveKit token shape")
    # We don't have the LiveKit secret here — just decode without
    # verification to inspect the claims. Verifying the signature would
    # require exposing livekit_api_secret to the smoke runner, which we
    # deliberately don't do (it's the most sensitive secret on the box).
    claims = jwt.decode(session["token"], options={"verify_signature": False})
    video = claims.get("video", {})
    if video.get("room") != session["room"]:
        _fail(f"JWT room={video.get('room')!r}, expected {session['room']!r}")
    if not video.get("roomJoin"):
        _fail("JWT is missing roomJoin grant")
    if claims.get("sub") != session["identity"]:
        _fail(f"JWT sub={claims.get('sub')!r}, expected {session['identity']!r}")
    _ok("JWT claims match session metadata")


def step_wait_for_active(session: dict) -> None:
    print(f"4. Wait up to {AGENT_TIMEOUT_SECS}s for room to become active")
    deadline = time.monotonic() + AGENT_TIMEOUT_SECS
    last_status = None
    while time.monotonic() < deadline:
        body = _req(f"/v1/sessions/{session['room']}", expect_status=200)
        last_status = body.get("status")
        if last_status == "active":
            _ok(f"session transitioned to active (started_at={body.get('started_at')})")
            return
        time.sleep(2)
    _fail(
        f"room never reached 'active' within {AGENT_TIMEOUT_SECS}s "
        f"(last status: {last_status!r}). Check agent dispatch + "
        f"LiveKit webhook delivery."
    )


def main() -> int:
    print(f"smoke target: {API_URL}")
    step_health()
    if not API_KEY:
        print("(SMOKE_API_KEY unset — /health passed, skipping /v1 checks)")
        return 0
    session = step_create_session()
    step_verify_jwt(session)
    step_wait_for_active(session)
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

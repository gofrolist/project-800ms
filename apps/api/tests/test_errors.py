"""Error envelope + request_id propagation tests.

No DB required — these exercise the middleware + exception handlers via
an in-process ASGI client. Fast; runs in every CI invocation.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

import errors
from errors import APIError
from request_id import RequestIdMiddleware, current_request_id


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    errors.install(app)

    @app.get("/raise-api-error")
    def _raise_api_error():
        raise APIError(401, "unauthenticated", "No key")

    @app.get("/raise-http")
    def _raise_http():
        raise HTTPException(status_code=404, detail="Nope")

    @app.get("/raise-validation")
    def _raise_validation(x: int):  # query param forces pydantic coercion
        return {"x": x}

    @app.get("/raise-unhandled")
    def _raise_unhandled():
        raise RuntimeError("boom")

    @app.get("/current-rid")
    def _current_rid():
        return {"request_id": current_request_id()}

    return app


@pytest.fixture
def client(app):
    # raise_app_exceptions=False lets us verify the unhandled-exception
    # handler instead of having the test transport re-raise straight
    # through. In production (Uvicorn), the ASGI server always catches.
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_api_error_returns_envelope(client):
    r = await client.get("/raise-api-error")
    assert r.status_code == 401
    body = r.json()
    assert set(body["error"].keys()) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "unauthenticated"
    assert body["error"]["message"] == "No key"
    assert body["error"]["request_id"] == r.headers["X-Request-ID"]


async def test_http_exception_maps_to_canonical_code(client):
    r = await client.get("/raise-http")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_validation_error_includes_field_errors(client):
    r = await client.get("/raise-validation?x=notanint")
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert "errors" in body["error"]
    assert len(body["error"]["errors"]) >= 1


async def test_unhandled_exception_is_generic(client):
    r = await client.get("/raise-unhandled")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # Must NOT leak the "boom" message to the client.
    assert "boom" not in body["error"]["message"]


async def test_request_id_header_round_trips_inbound(client):
    inbound = "abc123def456ghi789"
    r = await client.get("/current-rid", headers={"X-Request-ID": inbound})
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == inbound
    assert r.json()["request_id"] == inbound


async def test_request_id_generated_when_absent(client):
    r = await client.get("/current-rid")
    assert r.status_code == 200
    rid = r.headers["X-Request-ID"]
    # ULID = 26 Crockford base32 chars
    assert len(rid) == 26
    assert r.json()["request_id"] == rid


async def test_api_error_rejects_unknown_code():
    # This is a dev-time guard so we don't ship rogue codes.
    with pytest.raises(ValueError, match="Unknown error code"):
        APIError(418, "teapot", "not a real code")

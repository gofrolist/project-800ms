"""Smoke tests for the docs / OpenAPI surface.

No DB required — these exercise the static spec + docs HTML. Run on every
CI invocation. Heavier integration checks (auth actually being enforced)
live in test_auth.py / test_v1_sessions.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestOpenAPISpec:
    def test_spec_is_served(self, client: TestClient) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["info"]["title"] == "project-800ms API"
        assert spec["info"]["version"]
        # OpenAPI 3.1 is what FastAPI emits by default — lock it in so
        # generated SDKs know which version of the spec they're reading.
        assert spec["openapi"].startswith("3.1")

    def test_declares_api_key_security_scheme(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        schemes = spec.get("components", {}).get("securitySchemes", {})
        assert "APIKeyHeader" in schemes, list(schemes.keys())
        scheme = schemes["APIKeyHeader"]
        assert scheme == {"type": "apiKey", "in": "header", "name": "X-API-Key"}

    def test_v1_sessions_applies_security(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        create = spec["paths"]["/v1/sessions"]["post"]
        security = create.get("security", [])
        # The ApiKeyHeader scheme must be referenced on protected endpoints.
        assert any("APIKeyHeader" in block for block in security), security

    def test_health_is_open(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        health = spec["paths"]["/health"]["get"]
        # /health takes no auth — no security requirement should appear.
        assert not health.get("security")

    def test_documents_error_envelope_on_401(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        create = spec["paths"]["/v1/sessions"]["post"]
        responses = create["responses"]
        assert "401" in responses
        envelope = responses["401"]["content"]["application/json"]["schema"]
        assert "error" in envelope["properties"]

    def test_tags_populated(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        tag_names = {t["name"] for t in spec.get("tags", [])}
        assert {"sessions", "system"}.issubset(tag_names)


class TestDocsPages:
    def test_swagger_served(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200
        assert "swagger" in r.text.lower()

    def test_redoc_served(self, client: TestClient) -> None:
        r = client.get("/redoc")
        assert r.status_code == 200
        assert "redoc" in r.text.lower()

    def test_scalar_reference_served(self, client: TestClient) -> None:
        r = client.get("/reference")
        assert r.status_code == 200
        assert "api-reference" in r.text
        assert "/openapi.json" in r.text

    def test_reference_not_in_openapi_spec(self, client: TestClient) -> None:
        """/reference is a UI route, not an API endpoint — shouldn't
        clutter the spec."""
        spec = client.get("/openapi.json").json()
        assert "/reference" not in spec["paths"]

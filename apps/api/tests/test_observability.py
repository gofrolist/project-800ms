"""JSON logging + Prometheus /metrics tests."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from main import app
from observability import (
    JsonFormatter,
    configure_json_logging,
    room_var,
    tenant_id_var,
    tenant_slug_var,
)
from rate_limit import limiter
from request_id import request_id_var


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Fresh /health IP bucket per test."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def client():
    return TestClient(app)


# ─── JSON formatter ───────────────────────────────────────────────────────


def _format_one(level: int, msg: str, **extra: object) -> dict[str, object]:
    """Run a record through JsonFormatter and decode the resulting JSON."""
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    line = JsonFormatter().format(record)
    return json.loads(line)


class TestJsonFormatter:
    def test_core_fields_present(self):
        out = _format_one(logging.INFO, "hello")
        assert set(out) >= {
            "ts",
            "level",
            "logger",
            "msg",
            "request_id",
            "tenant_id",
            "tenant_slug",
            "room",
        }
        assert out["level"] == "INFO"
        assert out["msg"] == "hello"

    def test_request_context_propagates(self):
        """ContextVars set before a log record is formatted must appear
        as JSON fields on that record."""
        request_id_var.set("01TEST")
        tenant_id_var.set("tenant-uuid-xxx")
        tenant_slug_var.set("acme")
        room_var.set("room-42")
        try:
            out = _format_one(logging.WARNING, "hit")
            assert out["request_id"] == "01TEST"
            assert out["tenant_id"] == "tenant-uuid-xxx"
            assert out["tenant_slug"] == "acme"
            assert out["room"] == "room-42"
        finally:
            request_id_var.set("-")
            tenant_id_var.set("-")
            tenant_slug_var.set("-")
            room_var.set("-")

    def test_missing_context_renders_sentinels(self):
        """Outside a request scope, the defaults are '-' sentinels, not
        an exception."""
        out = _format_one(logging.INFO, "startup")
        assert out["request_id"] == "-"
        assert out["tenant_id"] == "-"

    def test_exc_info_flattens_to_string_field(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="t",
                level=logging.ERROR,
                pathname=__file__,
                lineno=0,
                msg="failed",
                args=None,
                exc_info=sys.exc_info(),
            )
            line = JsonFormatter().format(record)
            out = json.loads(line)
            assert "exc_info" in out
            assert "ValueError: boom" in out["exc_info"]

    def test_extra_kwargs_bubble_up(self):
        """`logger.info("x", extra={"foo": 1})` — the foo key should
        survive to the JSON output."""
        out = _format_one(logging.INFO, "x", foo=42, bar="hi")
        assert out["foo"] == 42
        assert out["bar"] == "hi"


class TestConfigureJsonLogging:
    def test_installed_once_even_on_reconfigure(self):
        """configure_json_logging() is idempotent — reapplying it
        shouldn't stack handlers and emit duplicate lines."""
        configure_json_logging()
        first = len(logging.getLogger().handlers)
        configure_json_logging()
        second = len(logging.getLogger().handlers)
        assert first == second == 1


# ─── Prometheus /metrics endpoint ─────────────────────────────────────────


class TestMetrics:
    def test_metrics_endpoint_is_open(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_renders_counter_after_health_hit(self, client):
        client.get("/health")
        body = client.get("/metrics").text
        assert "http_requests_total" in body
        assert "http_request_duration_seconds" in body
        # /health was recorded at least once.
        assert 'path="/health"' in body

    def test_metrics_skips_self(self, client):
        """Scrape the /metrics endpoint itself — it should not appear as
        a labeled series (we'd inflate cardinality forever)."""
        client.get("/metrics")
        client.get("/metrics")
        body = client.get("/metrics").text
        assert 'path="/metrics"' not in body

    def test_unknown_path_does_not_explode(self, client):
        r = client.get("/definitely-not-a-route")
        assert r.status_code == 404
        body = client.get("/metrics").text
        # Some series got recorded for this request — we just need to
        # confirm the middleware didn't error out.
        assert "http_requests_total" in body

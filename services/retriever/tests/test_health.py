"""Unit tests for the readiness-probe LLM check.

Targets `_check_rewriter_cached()` in `services/retriever/health.py`.
The probe is a metadata GET against `{llm_base_url}/models`, NOT a chat
completion — calling it on a 30 s cadence against a free-tier provider
(Groq's 1000 RPD) used to burn ~2880 calls/day and silently 503 every
/retrieve call once the daily cap hit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import httpx
import pytest
import respx

from health import (
    _REWRITER_CACHE_TTL_FAIL,
    _REWRITER_CACHE_TTL_OK,
    _check_rewriter_cached,
    reset_rewriter_cache,
)

_LLM_BASE = "http://llm-test.local"
_MODELS_URL = f"{_LLM_BASE}/models"
_CHAT_URL = f"{_LLM_BASE}/chat/completions"


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Point the probe at a mock LLM host per-test, and clear the
    rewriter cache so each test starts from a known state."""
    monkeypatch.setenv("DB_URL", "postgresql+asyncpg://placeholder/placeholder")
    monkeypatch.setenv("LLM_BASE_URL", _LLM_BASE)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("REWRITER_MODEL", "default-model")
    from config import get_settings

    get_settings.cache_clear()
    reset_rewriter_cache()
    yield
    get_settings.cache_clear()
    reset_rewriter_cache()


@pytest.mark.asyncio
async def test_probe_uses_models_endpoint_not_chat_completions() -> None:
    """The whole point of the fix: don't hit the billed chat surface.

    Asserts the probe issues GET /models with the bearer token and
    consumes zero quota at /chat/completions.
    """
    with respx.mock(assert_all_called=False) as mock:
        models_route = mock.get(_MODELS_URL).respond(200, json={"data": [{"id": "default-model"}]})
        chat_route = mock.post(_CHAT_URL).respond(200, json={})

        ok = await _check_rewriter_cached()

        assert ok is True
        assert models_route.called
        assert models_route.calls.last.request.headers["authorization"] == "Bearer test-api-key"
        assert chat_route.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429])
async def test_auth_and_quota_errors_report_not_ready(status: int) -> None:
    """A revoked key (401/403), a wrong base URL (404), or a rate-limit
    (429) must surface as `not ready` so operators get paged."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_MODELS_URL).respond(status)

        ok = await _check_rewriter_cached()

    assert ok is False


@pytest.mark.asyncio
async def test_5xx_reports_not_ready() -> None:
    """Upstream LLM hard-failure → not ready. The retriever should let
    the orchestrator restart the container rather than silently degrade."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_MODELS_URL).respond(500)

        ok = await _check_rewriter_cached()

    assert ok is False


@pytest.mark.asyncio
async def test_2xx_reports_ready() -> None:
    """Anything in [200, 500) that isn't an auth/quota signal counts as ready."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_MODELS_URL).respond(204)

        ok = await _check_rewriter_cached()

    assert ok is True


@pytest.mark.asyncio
async def test_network_error_reports_not_ready() -> None:
    """A connect/timeout failure surfaces as not-ready without leaking
    asyncpg/httpx exception messages (which can carry credentials)."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(_MODELS_URL).mock(side_effect=httpx.ConnectError("kaboom"))

        ok = await _check_rewriter_cached()

    assert ok is False


@pytest.mark.asyncio
async def test_result_is_cached_within_ttl() -> None:
    """Two probes inside the OK-TTL window must produce one HTTP call.

    This is the headline cost fix — the docker healthcheck fires every
    10 s; without caching, that's a request every 10 s = 8640/day."""
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).respond(200, json={"data": []})

        first = await _check_rewriter_cached()
        second = await _check_rewriter_cached()

        assert first is True
        assert second is True
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_negative_result_is_cached_within_fail_ttl() -> None:
    """A failed probe must also be cached — within the fail TTL — so the
    healthcheck firing every 10 s doesn't spam the LLM with retries during
    a sustained outage. Pinned separately from the OK case because the
    fail-TTL was the regression vector before two-tier caching."""
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).respond(401)

        first = await _check_rewriter_cached()
        second = await _check_rewriter_cached()

        assert first is False
        assert second is False
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_concurrent_probes_collapse_to_one_request() -> None:
    """Cold-start thundering-herd: N concurrent /ready calls before the
    cache is warm must coalesce into a single LLM request via the lock.

    Uses an asyncio.Event to gate the first probe in-flight: the side_effect
    awaits the event before responding, ensuring all 8 coroutines pass the
    pre-lock cache check (cold) and queue at the lock. Without the
    `_rewriter_lock`, this configuration would issue 8 real requests; the
    `call_count == 1` assertion is therefore load-bearing for the lock.
    """
    release = asyncio.Event()

    async def gated_response(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json={"data": []})

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).mock(side_effect=gated_response)

        async def kick_off() -> bool:
            return await _check_rewriter_cached()

        tasks = [asyncio.create_task(kick_off()) for _ in range(8)]
        # Yield control so every task reaches the lock-acquire await before
        # the first one is allowed to complete its HTTP call.
        for _ in range(8):
            await asyncio.sleep(0)
        release.set()

        results = await asyncio.gather(*tasks)

        assert all(results)
        assert route.call_count == 1


def test_cache_ttls_are_set_to_safe_defaults() -> None:
    """Pin the two-tier TTL values so a future change can't silently
    re-introduce the quota-burn issue (OK TTL too low) or the
    transient-failure-poisoning issue (FAIL TTL too high) without a
    failing test forcing a deliberate revisit. Plain sync assertions —
    no event loop required."""
    assert _REWRITER_CACHE_TTL_OK == 300.0
    assert _REWRITER_CACHE_TTL_FAIL == 60.0


@pytest.mark.asyncio
async def test_cache_expires_after_ok_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the OK-TTL window, the next probe re-hits the LLM.

    Patches `time.monotonic` rather than sleeping — the test must stay
    fast and not depend on real wall-clock time."""
    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr("health.time.monotonic", fake_monotonic)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).respond(200, json={"data": []})

        fake_now[0] = 0.0
        await _check_rewriter_cached()
        fake_now[0] = _REWRITER_CACHE_TTL_OK - 1.0
        await _check_rewriter_cached()  # still cached
        fake_now[0] = _REWRITER_CACHE_TTL_OK + 1.0
        await _check_rewriter_cached()  # cache expired → re-probe

        assert route.call_count == 2


@pytest.mark.asyncio
async def test_cache_expires_after_fail_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe re-fires after the (shorter) fail TTL — the whole
    point of the two-tier design. A 60 s outage of the LLM should not
    manifest as 5 minutes of degraded readiness."""
    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr("health.time.monotonic", fake_monotonic)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).respond(429)

        fake_now[0] = 0.0
        await _check_rewriter_cached()
        fake_now[0] = _REWRITER_CACHE_TTL_FAIL - 1.0
        await _check_rewriter_cached()  # still cached as failure
        fake_now[0] = _REWRITER_CACHE_TTL_FAIL + 1.0
        await _check_rewriter_cached()  # fail TTL expired → re-probe

        assert route.call_count == 2


@pytest.mark.asyncio
async def test_failure_does_not_extend_to_ok_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: if the cache used a single TTL again, a failed
    probe would stay cached for 5 min and this test would catch it.
    Probing at fail_ttl+1 must re-fire even though we're still well
    under ok_ttl."""
    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr("health.time.monotonic", fake_monotonic)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(_MODELS_URL).respond(500)

        fake_now[0] = 0.0
        await _check_rewriter_cached()
        # Halfway between fail TTL and ok TTL — must NOT be a cache hit.
        fake_now[0] = (_REWRITER_CACHE_TTL_FAIL + _REWRITER_CACHE_TTL_OK) / 2.0
        await _check_rewriter_cached()

        assert route.call_count == 2

---
title: "XFF-spoof bypass and shared-cache eviction in FastAPI IP rate limiter"
date: 2026-04-20
module: apps/api/rate_limit
component: authentication
category: security-issues
problem_type: security_issue
severity: critical
tags:
  - rate-limiting
  - x-forwarded-for
  - trusted-proxy
  - fastapi
  - token-bucket
  - timing-attack
  - cidr-validation
  - cache-eviction
related_components:
  - tooling
  - development_workflow
symptoms:
  - "Any Docker-network peer whose client IP begins with '172.' (covering all of 172.0.0.0/8, not just RFC1918 172.16.0.0/12) could spoof X-Forwarded-For and rotate through synthetic IPs to bypass the 60/min admin rate limit"
  - "A single shared 4096-slot TTLCache held both tenant-UUID buckets and IP buckets, so an attacker flooding >4096 distinct IPs could LRU-evict legitimate tenants' rate-limit state and weaken the pre-existing tenant limit the IP limit was meant to reinforce"
  - "Webhook 401 timing distinguished missing-auth-header from bad-signature branches: missing-header returned after one log line while bad-signature ran JWT verification, leaking which failure mode callers had triggered"
  - "rate_limited APIError responses were emitted without a Retry-After header, so well-behaved clients had no machine-readable backoff hint and automation tripped the 60/min admin cap from shared NAT egress IPs"
  - "No tests asserted the 61st admin request returns 429, that two distinct XFF values get independent buckets, or that admin IP rate-limit fires before _require_admin — a security-critical path had zero regression coverage"
root_cause: missing_validation
resolution_type: code_fix
---

# XFF-spoof bypass and shared-cache eviction in FastAPI IP rate limiter

## Problem

A prior security patch (commit `f6cc2f7`) added IP-keyed rate limits to `/v1/admin/*` and `/v1/livekit-webhook` to harden two unauthenticated surfaces, but the implementation reused a loose string-prefix proxy-trust check and co-located IP buckets in the same LRU cache as tenant-scoped buckets — creating an XFF-spoof bypass and a cache-eviction amplification path. It also shipped with zero test coverage on the new security paths, a unified-but-not-timing-equal webhook 401, and several smaller defects the ce-code-review flagged as P1–P3. Commit `101dbfa` fixes the composition failures, pins the invariants with tests, and promotes the knobs to Settings.

## Symptoms

What the reviewer / an attacker / an operator could observe on the pre-`101dbfa` code:

**Bypass and amplification (P1, critical):**

- **P1-1 XFF-spoof bypass + shared-cache eviction.** `_real_ip` honoured `X-Forwarded-For` whenever `request.client.host.startswith("172.")` — matching the entire `172.0.0.0/8` block (not just RFC1918 `172.16.0.0/12`) and, worse, doing a string prefix check instead of network membership. Any attacker reaching the API from an IP whose dotted-quad began with `172.` could rotate XFF per request to mint a fresh rate-limit key. A single 4096-entry `TTLCache` held both per-tenant buckets *and* per-IP buckets, so an XFF-flooding attacker could LRU-evict authenticated tenants' rate-limit state.
- **P1-2 Zero test coverage** on ordering (rate-limit before auth), bucket isolation per-IP, 429 enforcement, unified 401 body parity, CORS preflight, and the OpenAPI UGC note. The security-critical paths were unverified.

**Partial coverage (P1–P2):**

- **P1-3 Webhook 429 wedges session state.** If `/v1/livekit-webhook` drops a `room_started` event due to a 429, the later `room_finished` path silently stamps `audio_seconds=0` — indistinguishable from a legitimate zero-second session in billing/analytics.
- **P2-4 Timing side-channel on webhook 401.** The unified 401 body closed the lexical channel but not the latency channel: the missing-header branch skipped `_receiver.receive()` entirely, so its response was measurably faster than the bad-signature branch. Response-body parity alone is not side-channel closure.

**Configuration + observability (P2–P3):**

- **P2-5 Admin 60/min too tight, drains on 503.** Rates were hardcoded — no knob for ops scripts bursting from one egress (Terraform `for_each`, CI seeding, bulk key rotation). Worse, when `admin_api_key` was empty (admin surface disabled, all routes 503), the IP bucket was still consumed, punishing legitimate operators on shared egress and letting attackers probe a disabled endpoint to thrash the cache.
- **P3-6** Parameter name `tenant_id` misleading after being reused for IP keys.
- **P3-7** Stale admin module docstring — no mention of the new rate-limit dep or disabled-surface behaviour.
- **P3-9** Missing `Retry-After` header on 429 — programmatic clients (Terraform providers, CI runners, the LiveKit retry loop) had nothing to key their backoff on.
- **P3-11** Test fixtures didn't reset IP buckets between tests — order-dependent flakes waiting to happen.
- **P3-12** Duplicate `from fastapi import` block in `webhooks.py`.

## What Didn't Work

The initial fix (`f6cc2f7`) did *add* the defense layer — IP limits were being enforced in isolation — but the layer's composition with existing security was weak in ways that partially undermined what was already there.

The shared-cache design was not an oversight; it was inherited from a time when the assumption held. (session history) The per-tenant token-bucket code had been rewritten on 2026-04-19 to replace slowapi's `@limit` decorator (which couldn't accept dynamic per-tenant rates), and at that point `_buckets` only held tenant UUIDs — 4096 slots was more than enough for the expected tenant count. The problem only emerged once `f6cc2f7` added IP-keyed buckets into the same cache on 2026-04-20. The two threat profiles (~hundreds of auth'd tenants vs. potentially thousands of distinct IPs) met in a slot pool sized for only one of them.

The trust heuristic had the same shape of problem: `_real_ip`'s `startswith("172.")` was written for the Docker bridge (typically 172.17.0.0/16) and was *approximately* correct when the only caller was `/health` under slowapi — the blast radius of a spoofed `/health` limit is near-zero. (session history) Promoting that heuristic to drive security-critical admin and webhook buckets turned an approximate check into an exploitable one.

**The string-prefix XFF trust was actively dangerous as a rate-limit input.**

```python
# f6cc2f7 — rejected
def _real_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    # Bug: "172.".startswith() matches 172.0.0.0/8 (public + private),
    # and string prefix is not network membership in any case.
    if xff and client_host.startswith("172."):
        return xff.split(",")[0].strip()
    return get_remote_address(request)
```

Because rate-limit keys are *derived* from this function's return value, any attacker that could control XFF (which is any attacker, since it's a client-supplied header) could rotate the rate-limit key per request and never exhaust a bucket. The layer meant to cap abuse passively accepted attacker-supplied input as its cache key.

**The shared-cache design made the new layer an amplification vector for the old layer.**

A single `TTLCache(maxsize=4096)` held both per-tenant buckets (authoritative identity) and per-IP buckets (weak identity). An unauthenticated IP flood could LRU-evict authenticated tenants' rate-limit state — meaning defense-in-depth for unauth paths *weakened* rate limiting for auth paths. Different threat profiles, same cache slots, same eviction pool.

**Lexical response parity is necessary but not sufficient for side-channel closure.**

`f6cc2f7` unified the 401 body string on the webhook so an attacker couldn't distinguish "missing header" from "bad signature" by reading the response. But the two branches did different amounts of work: the missing-header branch short-circuited before `_receiver.receive()`, so its response was measurably faster. An attacker timing the response could still distinguish the two branches.

**Hardcoded rates are a prod operations hazard.**

60/min for admin is fine for a human. It 429s a Terraform `for_each` rotating 80 keys or a CI run seeding 200 test tenants. The knob needed to exist before those workflows hit it, not after.

## Solution

The complete fix lives mostly in `apps/api/rate_limit.py` with Settings additions and narrow patches to the routes and error handler.

### 1. Real CIDR membership for proxy trust

`apps/api/rate_limit.py`:

```python
def _is_trusted_proxy(client_host: str) -> bool:
    """True when the direct TCP peer sits inside a configured proxy CIDR."""
    if not client_host:
        return False
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    for cidr in settings.trusted_proxy_cidrs:
        try:
            if client_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def _real_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    client_host = request.client.host if request.client else ""
    if xff and _is_trusted_proxy(client_host):
        return xff.split(",")[0].strip()
    return get_remote_address(request)
```

`settings.trusted_proxy_cidrs` defaults to `["172.16.0.0/12"]` — the actual RFC1918 Docker-bridge block, not the `172.0.0.0/8` superset. Misconfigured CIDRs are caught per-entry; the function is total and never raises into a request handler.

### 2. Split caches by namespace

```python
# Two separate caches so an unauth-IP flood can't LRU-evict authenticated
# tenant rate-limit state. Tenant cache sized for expected max live tenants;
# IP cache is larger because webhook / admin traffic legitimately spans
# many distinct IPs.
_tenant_buckets: TTLCache[str, _TokenBucket] = TTLCache(maxsize=4096,  ttl=600)
_ip_buckets:     TTLCache[str, _TokenBucket] = TTLCache(maxsize=32768, ttl=600)
_buckets_lock = Lock()


def _get_bucket(cache: TTLCache[str, _TokenBucket], key: str, rate_per_minute: int) -> _TokenBucket:
    with _buckets_lock:
        bucket = cache.get(key)
        if bucket is None or bucket.capacity != rate_per_minute:
            bucket = _TokenBucket.for_rate(rate_per_minute)
            cache[key] = bucket
        return bucket
```

`_get_bucket` now takes the cache as a parameter (P3-6: renamed `tenant_id` → `key`), so tenant and IP callers pass their own cache and share nothing. One lock covers both caches for test-teardown simplicity.

### 3. Timing-parity dummy JWT on the missing-header branch

`apps/api/routes/webhooks.py`:

```python
# Pre-built JWT deliberately signed with a bogus key so _receiver.receive()
# parses the header + payload, runs HMAC verification, and raises. Same
# computational path as a real bad-signature request, so calling it on
# the missing-header branch gives the two 401 responses matching latency
# profiles — closing the timing side-channel that a unified error string
# alone cannot close.
_TIMING_PARITY_DUMMY_JWT = jwt.encode(
    {"iss": "timing-parity-placeholder", "exp": 9_999_999_999, "sha256": ""},
    "x" * 32,  # deliberately wrong — verification will fail
    algorithm="HS256",
)

# ...inside livekit_webhook()...
if not authorization:
    try:
        _receiver.receive(raw_body, _TIMING_PARITY_DUMMY_JWT)
    except Exception:
        # Expected — the dummy is signed with a bogus key. We want the
        # crypto work, not the result.
        pass
    logger.warning("Webhook: missing Authorization header")
    raise APIError(401, "unauthenticated", "Webhook authentication failed")
```

The raw body is read up-front on both branches so body-size-dependent work (SHA-256 over the payload) runs identically. Both branches now have lexically identical response bodies *and* equivalent work — closing both the content and timing channels.

> **Note (session history):** The initial walkthrough recommendation for Finding 4 was to Acknowledge-and-document the timing gap rather than fix it, on the reasoning that the signal only tells an attacker whether their Authorization value reached the verifier. The recommendation was reversed when the user asked for the full fix. The dummy-verify approach is a stronger closure than the initial recommendation suggested — worth the ~1ms per missing-header 401 to make the two branches actually match.

### 4. Admin short-circuit when disabled

```python
async def enforce_admin_ip_rate_limit(request: Request) -> None:
    """IP-based rate limit for /v1/admin/*.

    Short-circuits when `settings.admin_api_key` is empty — the admin
    surface is already disabled in that case (every route returns 503
    via `_require_admin`), so consuming an IP-bucket token would only
    punish legitimate operators on a shared egress IP and let attackers
    thrash the cache by probing a disabled endpoint.
    """
    if not settings.admin_api_key:
        return
    ip = _real_ip(request)
    bucket = _get_bucket(_ip_buckets, f"admin-ip:{ip}", settings.admin_ip_rate_per_minute)
    if not bucket.consume():
        raise APIError(429, "rate_limited", "Rate limit exceeded")
```

### 5. Settings-tunable rate knobs

`apps/api/settings.py`:

```python
# IP-based rate limit for /v1/admin/* (defense-in-depth against online
# brute-force of admin_api_key). 60/minute is generous for a human
# operator and still infeasible against a 256-bit key. Raise for ops
# workflows that burst from one egress (Terraform for_each, CI seeding,
# bulk key rotation) — e.g. 300 or 600.
admin_ip_rate_per_minute: int = 60

# IP-based rate limit for /v1/livekit-webhook. 1000/minute is well above
# normal LiveKit event rates from a single egress; raise if your deploy
# runs many concurrent rooms behind one LiveKit instance.
webhook_ip_rate_per_minute: int = 1000

# CIDR ranges whose TCP peer we trust to pass a real-client IP via XFF.
# Default: RFC1918 172.16.0.0/12 — the Docker bridge range in our compose
# deploy. Empty list disables XFF trust entirely.
trusted_proxy_cidrs: list[str] = ["172.16.0.0/12"]
```

Wired through `docker-compose.yml` so ops can tune without a rebuild.

### 6. `Retry-After: 60` on every 429

`apps/api/errors.py`:

```python
async def api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    headers: dict[str, str] | None = None
    if exc.code == "rate_limited":
        # Tell programmatic clients (Terraform providers, CI runners, LiveKit
        # webhook retry) when to try again. 60s is a conservative upper bound —
        # a bucket refills in <= 60s for any configured rate, so a client that
        # waits this long is guaranteed at least one token back.
        headers = {"Retry-After": "60"}
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.detail, exc.extra),
        headers=headers,
    )
```

### 7. Test coverage pinning the invariants

`apps/api/tests/test_rate_limit.py` adds six tests that turn the security properties into executable specs — ordering (rate limit before auth check), per-IP bucket isolation via XFF, 429 enforcement on both endpoints, `Retry-After` presence, and the disabled-admin short-circuit. Bucket reset is now in the shared `override_db` fixture via `_reset_buckets_for_tests()` so tests don't pollute each other (P3-11).

Example — the ordering invariant:

```python
async def test_admin_ip_rate_limit_fires_before_admin_key_check(client, enable_admin):
    """Bad X-Admin-Key requests still consume the admin-ip bucket.
    The router-level `enforce_admin_ip_rate_limit` must run before
    `_require_admin`, otherwise an attacker can brute-force the admin key
    at unlimited rate."""
    object.__setattr__(settings, "admin_ip_rate_per_minute", 60)
    _reset_buckets_for_tests()
    headers = {"X-Admin-Key": "not-the-real-key"}
    for i in range(60):
        r = await client.get("/v1/admin", headers=headers)
        assert r.status_code == 401  # auth fails but bucket consumes
    r = await client.get("/v1/admin", headers=headers)
    assert r.status_code == 429    # 61st request: bucket empty
```

Test count: 115 → 123.

### 8. Webhook-drop observability (P1-3 partial)

Rather than add a reconciler before we know the 429 rate in prod, the `room_finished` fallback now logs a pointer to the `http_requests_total{status="429",path="/v1/livekit-webhook"}` metric:

```python
logger.warning(
    "Webhook room_finished incomplete room=%s tenant=%s: "
    "no room_started observed and event.room.creation_time "
    "missing; audio_seconds=0 may be inaccurate. Check for "
    'dropped webhooks (see http_requests_total{status="429"} '
    "for path=/v1/livekit-webhook).",
    room_name, session.tenant_id,
)
```

If the 429 rate ever goes nonzero in prod we'll know where to look; full reconciliation is deferred until there's signal. (session history) The decision to defer was explicit — building a cron sweeper + LiveKit room-API reconciler before observing real 429 rates risks overengineering a problem that may not exist under real load.

## Why This Works

**P1-1 (XFF bypass + eviction).** Two linked defects, one root cause each:

1. *String prefix != network membership.* Using `ipaddress.ip_network(cidr).contains(ip)` makes the check a real CIDR test. `"172.0.0.0".startswith("172.")` is true; so is `"172.250.1.1".startswith("172.")` — the latter is public space. The fix also makes the trust list *configurable* so non-Docker deploys (ALB, bare-metal proxy) can express their real topology.
2. *Shared cache between namespaces with different threat profiles.* Per-tenant buckets have ~hundreds of distinct keys (bounded by tenant count); per-IP buckets can legitimately see thousands of distinct keys (webhook sources, admin egress IPs, CI runners). Putting both in a 4096-entry LRU meant an adversary controlling one could evict the other. Splitting into `_tenant_buckets` (4096) and `_ip_buckets` (32768) gives each namespace its own eviction pool sized for its own legitimate cardinality.

**P1-2 (zero coverage).** Security properties that aren't expressed as executable assertions aren't properties. The six new tests pin: *ordering* (rate limit runs before auth), *isolation* (distinct XFF → distinct buckets), *enforcement* (429 at the configured limit), *response shape* (Retry-After present, envelope code is `rate_limited`), and the *negative* case (short-circuit when admin is disabled). Each test fails if a future refactor silently breaks the invariant.

**P2-4 (timing channel).** Unifying the response body is a lexical fix for a lexical channel. Timing is a separate channel and requires a separate fix: the two branches must do equivalent work. Running `_receiver.receive(raw_body, _TIMING_PARITY_DUMMY_JWT)` on the missing-header path forces HMAC-SHA256 over a dummy of the same structural shape as a real JWT, and — crucially — reading `raw_body` up-front means the SHA-256-over-payload cost is identical on both branches regardless of body size. This won't survive a sufficiently well-instrumented adversary with perfect network timing (CPU scheduling jitter plus network variance gives slack), but it removes the >100µs gap between "skip crypto" and "run crypto" that made the channel trivially exploitable from a shared cloud network.

**P2-5 (admin short-circuit + tunability).** When the admin surface is disabled, every request is going to 503 anyway. Consuming a token from the IP bucket in that case is pure downside: legitimate ops traffic hitting 60/min gets 429 instead of the diagnostic 503 ("admin is off"), and adversaries probing a disabled endpoint can still thrash the bucket cache. The short-circuit preserves the diagnostic signal and denies the attacker free cache churn. The Settings knobs (`admin_ip_rate_per_minute`, `webhook_ip_rate_per_minute`) let ops widen limits for automation bursts without redeploying code — which matters because the rate limit is *defense-in-depth*, not the actual access boundary (`secrets.compare_digest` is).

**P3-9 (Retry-After).** A 429 without `Retry-After` forces clients to guess backoff. LiveKit's webhook retry, Terraform providers, and CI runners all honour the header if present. The `errors.py` handler attaches it on any `rate_limited` APIError, so every rate-limited path — tenant bucket or IP bucket — gets it for free without per-call-site plumbing.

**General principles extracted from this incident:**

1. *Defense-in-depth layers must not weaken existing security.* A layer that uses attacker-controlled input as a cache key, or that shares a finite resource (cache slots) with a stronger layer, can actively undermine the property it was meant to reinforce.
2. *Shared caches between namespaces with different threat profiles are amplification vectors.* Different cardinalities, different adversary control, different eviction tolerance — separate them.
3. *Lexical response parity is necessary but not sufficient for side-channel closure.* Equal bytes out, equal work done, equal reads from the same inputs — all three or none.
4. *Security knobs must be operable.* If ops can't widen a defense layer for legitimate bursts without a code deploy, they'll disable it entirely or work around it.

## Prevention

Concrete practices to apply on future unauth-surface hardening work:

1. **Use `ipaddress.ip_network` for proxy trust, never string prefixes.** `startswith()` on a dotted-quad is a syntactic match, not a semantic one — it will match public-IP supersets, will not handle IPv6, and cannot express non-power-of-256 subnet masks. Same rule for any "is this IP in my trusted set" check.

2. **Separate caches by namespace when threat profiles differ.** If `auth_id → state` and `ip → state` share a TTLCache, an unauthenticated flood can evict authenticated identities. Size each cache for the expected cardinality of its namespace and let them evict independently.

3. **Pin security invariants with tests at the HTTP layer.** Don't rely on "I ordered the dependencies right" — write a test that makes 60 bad-auth requests and asserts the 61st is 429. Don't rely on "I unified the response body" — assert `response.json()["error"]["code"] == "rate_limited"` and `response.headers["Retry-After"] == "60"`. The tests that matter here are: *ordering* (dep A runs before dep B), *isolation* (bucket per key, not shared), *enforcement* (429 at configured limit, not above), *response shape* (body and headers), *disabled-surface behaviour* (layer doesn't run when underlying feature is off). See `apps/api/tests/test_rate_limit.py` for a working template.

4. **Constant-time branch parity means equal work, not just equal responses.** If two branches produce the same response body, an attacker timing latency can still distinguish them. For crypto-verification paths specifically: read request inputs at the same point, run the same verification call on both branches (with a pre-built dummy on the "skip" branch if needed), raise identical exceptions. Acknowledge up-front this is mitigation, not elimination — adversaries with perfect network timing still win eventually.

5. **Make security knobs Settings-tunable from day one.** Anything an operator might need to widen under legitimate load (rate caps, trusted CIDR lists, feature toggles) belongs in `settings.py`, not as a magic number in the dep function. Rule of thumb: if the value appears in a log message or a 429 response body, it should be a setting. Pipe through `docker-compose.yml` and/or Terraform SSM so it can be changed without a code deploy.

6. **Attach `Retry-After` to every 429 centrally, not per-call-site.** Do it in the error handler that owns the envelope (`errors.py` here). Programmatic clients — retry loops, infrastructure providers, webhook senders — key their backoff on it. Missing it means they guess, and guesses are either too fast (re-429s) or too slow (dropped work).

7. **Defense-in-depth layers must short-circuit on already-disabled paths.** If the feature behind the layer is off (`admin_api_key == ""` → every route 503), the layer should not consume cache slots / tokens / any bounded resource. Otherwise the layer becomes a free DoS amplifier against itself — adversaries probing a disabled endpoint thrash the cache, legitimate operators on shared egress see 429 instead of the 503 that would tell them what's actually wrong.

8. **When an invariant is hard to maintain, express it as a test, not a comment.** The comment "this dep must run before the admin-key check" rots; the test that asserts `60 × 401 → 1 × 429` does not.

9. **Use documented commit scopes (`api, agent, web, infra, ci`) — not `security` or other ad-hoc scopes.** (session history) The follow-up commit `101dbfa` used `fix(api,security): …` which violates the project convention documented in `CLAUDE.md`; the scope field is intended for the module/component, not the risk category (which belongs in the body). Going forward, `fix(api): address ce-code-review follow-ups` is the right shape — severity lives in the commit body.

## Related Prior Art

Not solution docs (none existed before this one — this is the first entry under `docs/solutions/`), but earlier planning and brainstorming surfaces that establish context:

- **`docs/brainstorms/2026-04-19-voice-stack-ab-experiment-requirements.md`** — originates the "rate-limit dep prevents enumeration oracle" pattern for tenant-scoped routes. The IP rate-limit deps added in `f6cc2f7`/`101dbfa` are the unauthenticated-endpoint analog of the same defense-in-depth pattern.
- **`docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md`** — repeatedly references `enforce_tenant_rate_limit` as the existing pattern to reuse. `101dbfa` is explicitly *not* extending that (it adds IP-keyed, not tenant-keyed, limiters) — worth calling out the contrast so future readers don't conflate the two mechanisms. This plan also contains a stale file-location pointer (`apps/api/auth.py::enforce_tenant_rate_limit` — actual location is `apps/api/rate_limit.py`) that predates this work but should be refreshed.

## Follow-ups

- **Webhook reconciler** — build a sweeper for `sessions.status != 'ended'` + `started_at < now() - Nmin` that queries LiveKit room-state API, *only if* production metrics show sustained webhook 429s.
- **Doc refresh** — `docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md:430` references the wrong file for `enforce_tenant_rate_limit`; `CLAUDE.md:36` describes the pre-101dbfa XFF-trust mechanism. A targeted `/ce-compound-refresh` scoped to the rate-limit module layout would catch both.

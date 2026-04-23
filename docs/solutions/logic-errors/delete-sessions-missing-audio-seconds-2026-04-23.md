---
title: "DELETE /v1/sessions skipped audio_seconds computation causing silent billing under-count"
module: apps/api
date: 2026-04-23
category: docs/solutions/logic-errors
problem_type: logic_error
component: service_object
severity: critical
root_cause: missing_workflow_step
resolution_type: code_fix
symptoms:
  - DELETE-closed sessions have audio_seconds=NULL despite non-null started_at/ended_at
  - GET /v1/usage shows 0 audio contribution from UI-initiated closes while webhook-closed sessions report correctly
  - No error surfaces at runtime — DELETE returns 200, webhook returns 204, logs clean
  - Existing DELETE tests passed because they asserted status + ended_at but never audio_seconds
  - Webhook retries after DELETE short-circuit because ended_at is no longer None
tags:
  - livekit
  - session-lifecycle
  - idempotency
  - webhook-interaction
  - audio-seconds
  - billing
  - delete-endpoint
---

# DELETE /v1/sessions skipped audio_seconds computation causing silent billing under-count

## Problem

The `DELETE /v1/sessions/{room}` endpoint in `apps/api/routes/sessions.py` (introduced in commit `600532a`) marked sessions terminal — `status='ended'`, `ended_at=now` — but never populated `audio_seconds`. Only the `room_finished` webhook computed that field, and its idempotency guard (`if session.ended_at is None:`) short-circuited once DELETE had already committed. Since `GET /v1/usage` coalesces `NULL → 0` for billing aggregation, every UI-initiated hang-up was silently under-billed to zero.

## Symptoms

- Every session closed via `DELETE /v1/sessions/{room}` landed in the `sessions` table with `audio_seconds = NULL`, despite non-null `started_at` and `ended_at`.
- `GET /v1/usage` reports showed `audio_seconds = 0` contribution from user-initiated closes; sessions closed via LiveKit `empty_timeout → room_finished` correctly reported duration.
- No error at runtime: DELETE returned 200 with session details, the webhook returned 204, logs were clean. The bug was invisible to HTTP monitoring.
- The six DELETE tests in `apps/api/tests/test_v1_sessions.py` all passed — they asserted `response.status_code`, `status == "ended"`, and `ended_at is not None` but never the value of `audio_seconds`.
- Webhook retries after a DELETE saw `session.ended_at is not None` (DELETE set it) and skipped the `audio_seconds` branch every time. No self-healing path existed.

## What Didn't Work

- **The original code looked complete.** The DELETE handler's shape — `session.status = "ended"; session.ended_at = now; await db.flush(); await db.commit()` — mirrored the webhook's happy-path writes. No visible missing branch, no TODO, no commented-out line. The bug was a missing *write*, not wrong logic on a present line.
- **Webhook-only was the original design; DELETE was added without revisiting it.** (session history) Commit `33a09c7` from April 18 built `/v1/livekit-webhook` assuming `room_finished` was the sole session-close path. Usage aggregation in `apps/api/routes/usage.py` was built around that assumption — the `coalesce(sum(audio_seconds), 0)` at line 97 was added defensively for sessions interrupted before `room_started` (no `started_at`), NOT to mask NULLs on legitimately completed sessions. When DELETE was introduced as a UI hang-up convenience, the webhook-only assumption was never revisited.
- **Assuming the webhook would back-fill was wrong.** The webhook's `if session.ended_at is None:` guard is correct idempotency against replayed events, but it also meant once DELETE wrote `ended_at`, every subsequent `room_finished` (which *does* fire — DELETE tears down the LiveKit room) was treated as a duplicate and skipped the `audio_seconds` compute.
- **Python-level `if session.status != 'ended'` guards hide concurrent races.** An earlier read-then-write shape would let two concurrent DELETEs both pass the check and clobber each other's `ended_at`, and would not converge with a racing webhook write. The fix had to push idempotency into the SQL layer.

## Solution

Commit `07ca8ef` moves the `audio_seconds` computation into the DELETE handler itself and uses an atomic conditional `UPDATE ... WHERE status=<current>` that doubles as a concurrent-DELETE + webhook-race guard.

**Before** (`apps/api/routes/sessions.py`, commit `600532a`):
```python
await _delete_livekit_room(room)

# DB close — no audio_seconds write.
session.status = "ended"
session.ended_at = datetime.datetime.now(datetime.UTC)
await db.flush()
await db.commit()
return _session_to_details(session)
```

**After** (`apps/api/routes/sessions.py`, commit `07ca8ef`):
```python
await _delete_livekit_room(room)

# Conditional UPDATE guards against concurrent DELETE races: if two
# requests both pass the status check above, only one row update
# lands ``status='ended'``; the second finds 0 affected rows and
# returns the already-closed state below.
now = datetime.datetime.now(datetime.UTC)
audio_seconds = (
    max(int((now - session.started_at).total_seconds()), 0)
    if session.started_at is not None
    else None
)
update_stmt = (
    SessionRow.__table__.update()
    .where(SessionRow.id == session.id)
    .where(SessionRow.status == session.status)   # race guard
    .values(status="ended", ended_at=now, audio_seconds=audio_seconds)
)
result = await db.execute(update_stmt)
await db.commit()

if result.rowcount == 0:
    # Another concurrent DELETE (or the room_finished webhook)
    # beat us to the update. Re-read so the response reflects the
    # winning write's timestamps rather than our local mutations.
    await db.refresh(session)
    return _session_to_details(session)

# Reflect the UPDATE in the in-memory session object so the response
# body uses the values we actually wrote.
session.status = "ended"
session.ended_at = now
session.audio_seconds = audio_seconds
return _session_to_details(session)
```

Pinned by `apps/api/tests/test_v1_sessions.py::test_delete_session_ends_active_session`, which now asserts `body["audio_seconds"] is not None` in addition to the previously-existing `status` and `ended_at` assertions.

## Why This Works

Two handlers write terminal state to the same row, but only one computed `audio_seconds`. The fix makes the invariant explicit: `audio_seconds` belongs to "the final state of the session row," and whichever handler wins the terminal-state race must populate it. Both handlers now compute the same formula (`max(int((ended_at - started_at).total_seconds()), 0)`).

(session history) Three fix options were weighed during the walk-through — change the webhook guard to `if audio_seconds is None`, have DELETE not set `ended_at` at all and let the webhook own terminal state, or compute `audio_seconds` inside DELETE. The third was chosen because it collapsed two concerns (audio_seconds gap + concurrent-DELETE race from adversarial finding `adv-4`) into a single atomic UPDATE without changing the synchronous DELETE response contract.

The conditional `WHERE status=<current>` clause is doing double duty:

1. **Idempotency against concurrent DELETEs.** Two requests both pass the Python-level `if session.status != 'ended'` check (they read the same stale row), but only one UPDATE lands — the second sees `rowcount == 0` and re-reads the winner's timestamps. Keeps `ended_at` stable across retries, which matters for billing joins.
2. **Race convergence with the webhook.** If `room_finished` fires between our SELECT and our UPDATE, the webhook flips `status` to `ended`; our UPDATE's `WHERE status=<stale>` then matches 0 rows and we re-read the webhook's (correct) `audio_seconds` instead of clobbering it.

The `session.audio_seconds = audio_seconds` write on the in-memory object after a successful UPDATE keeps the response body consistent with what was committed.

**The `started_at=None` edge case is intentionally preserved.** If DELETE fires on a pending session, `started_at` is NULL and `audio_seconds` stays NULL — no audio was exchanged. In commit `07ca8ef`, DELETE on `status='pending'` returns 409 outright, so the NULL-audio branch is defensive rather than an expected path.

## Prevention

1. **When adding a new terminal-state code path, audit every OTHER terminal-state handler for fields they compute.** Concrete discovery pattern:
   ```bash
   rg "status\s*=\s*['\"]ended['\"]" apps/api/
   rg "status=\"ended\""              apps/api/
   rg "\.ended_at\s*="                apps/api/
   ```
   Any field written by one terminal handler must be written by every other, or the handlers produce different row shapes depending on who wins. If two handlers must stay symmetric, write a comment at each site naming the other.

2. **Happy-path tests for state-transition endpoints must assert every writable field, not just the status enum.** The original DELETE tests asserted `status` and `ended_at` but not `audio_seconds` — which is exactly the field that silently went NULL. Rule: if an endpoint writes N columns, the happy-path test asserts N columns. Do not rely on "the status transition implies the rest."

3. **Prefer SQL-layer idempotency (`UPDATE ... WHERE status=<current>`) over Python-layer pre-checks (`if session.status != 'ended': ...`).** The SQL form is naturally race-safe when multiple handlers can close the same row (DELETE + webhook retry + second DELETE from a retry-happy client). The Python form requires row locks (`SELECT ... FOR UPDATE`) to match it, which adds latency and deadlock surface.

4. **Defensive `coalesce(NULL, 0)` in billing queries masks genuine NULLs.** (session history) The `coalesce(sum(audio_seconds), 0)` in `apps/api/routes/usage.py:97` was added as defense against sessions interrupted before `room_started` fired — not to mask NULLs on completed sessions. When a defensive coalesce exists at the read layer, NULL values at the write layer become invisible in reports.

## Known Follow-ups

Code-review suggestions captured for future hardening (not blocking on this doc):

- **Extract a shared `_compute_audio_seconds(started_at, ended_at)` helper** imported by both `delete_session` (in `routes/sessions.py`) and `room_finished` (in `routes/webhooks.py`). Prevention item #1 names the symmetric-formula invariant, but today both sites re-implement it — the shared helper turns the invariant into something a type checker can enforce.
- **Prefer `sqlalchemy.update(SessionRow)` over `SessionRow.__table__.update()`** so the statement stays at the ORM layer. The Table-level construct bypasses ORM session state synchronization, which is why the happy-path currently needs the three manual attribute writes (`session.status = ...; session.ended_at = ...; session.audio_seconds = ...`) to keep the response body consistent.
- **Replace the happy-path manual reflection with `await db.refresh(session)`** to mirror the rowcount==0 branch. A single source of truth for "what got committed" eliminates the drift risk where a future column added to `.values(...)` silently skips the mirror write.
- **Strengthen the happy-path test** from `assert body["audio_seconds"] is not None` to `assert body["audio_seconds"] == expected_duration` (seeding `started_at = now - 42s` and asserting `42 ± 1`). Catches formula drift between DELETE and the webhook that the `is not None` assertion cannot.
- **Add a unit test for the `rowcount == 0` branch.** Stub `db.execute` to return `rowcount == 0`, assert the handler calls `db.refresh(session)` and returns the refreshed details. The race-safety claim is currently only argued in prose, never executed.

## Related Issues

- **[docs/solutions/security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md](../security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md)** — §P1-3 "Webhook 429 wedges session state" flagged the same `audio_seconds=0` billing symptom from a different angle (webhook 429 drops `room_started` → `room_finished` falls back to 0). That doc addressed it as a monitoring concern (log pointer + metric); this doc closes the DELETE-bypass path authoritatively. **The XFF doc's §8 should be refreshed** to note that DELETE-bypass is no longer an input to the zero-audio symptom after `07ca8ef` — only genuine 429-drops of `room_started` can still produce it.
- **[#34](https://github.com/gofrolist/project-800ms/issues/34)** (closed) — DELETE-before-join JWT ghost-room race. Same teardown surface, different symptom (caller auth, not usage accounting).
- **[#35](https://github.com/gofrolist/project-800ms/issues/35)** (closed) — startup warning when `LIVEKIT_URL` is unset. Sibling observability gap for the DELETE path.
- **[#36](https://github.com/gofrolist/project-800ms/issues/36)** (closed) — Prometheus counter for swallowed LiveKit failures. Sibling observability gap on the LiveKit side.
- **[#37](https://github.com/gofrolist/project-800ms/issues/37)** (closed) — agent-side double-`task.cancel` guard. Closest conceptual cousin: same "two writers racing on session teardown" shape, but agent-process-local rather than webhook-vs-DELETE.

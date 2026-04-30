---
title: "Process-local turn counter collides UNIQUE(session_id, turn_id) after agent restart"
module: services/agent/kb_retrieval
date: 2026-04-27
category: docs/solutions/logic-errors
problem_type: data_corruption
component: agent_processor
severity: high
root_cause: process_local_state_in_durable_uniqueness_key
resolution_type: code_fix
symptoms:
  - retrieval_traces INSERT raises IntegrityError on duplicate (session_id, turn_id)
  - The agent's broad `except Exception` routes every IntegrityError to the refusal-fallback path → 100% of post-restart turns become refusals for the rest of the session
  - No visible error to the caller — the user just hears the refusal prompt every time
  - Logs show repeated `kb_retrieval.call_failed kind=DbUnavailable` (or HTTPStatusError if the retriever's exception handler fires first)
  - Fresh sessions are unaffected; ONLY sessions that span an agent restart trip it
tags:
  - unique-constraint
  - session-state
  - agent-restart
  - monotonic-counter
  - idempotency
  - kb-retrieval
  - turn-id
---

# Symptom

`KBRetrievalProcessor` builds a `turn_id` from a process-local
counter. Each new processor instance (one per dispatched LiveKit
room) starts the counter at zero and emits `t-00001`, `t-00002`,
etc. The retriever writes one `retrieval_traces` row per turn,
constrained by `UNIQUE(session_id, turn_id)`.

When the agent container is replaced mid-session — rolling deploy,
OOM-kill, container restart — a NEW `KBRetrievalProcessor` boots up
in the SAME LiveKit room. It dispatches with the same `session_id`
and replays `t-00001` on the first turn after restart.

The retriever sees a duplicate `(session_id, turn_id)`, raises
`IntegrityError`, the retrieve handler converts it to
`DbUnavailable` 503, and the agent's catch-all `except Exception`
routes the response to the refusal fallback path.

The user perceives "the assistant suddenly stopped knowing
anything" until the call ends. There's no operator-facing alarm
because every individual log line looks like a routine transient
failure (503 → refusal cascade is documented and self-healing for
real DB outages).

# Root cause

```python
class KBRetrievalProcessor(FrameProcessor):
    def __init__(self, *, retriever_url, tenant_id, session_id, ...):
        self._turn_counter = 0  # process-local

    async def process_frame(self, frame, direction):
        self._turn_counter += 1
        turn_id = f"t-{self._turn_counter:05d}"  # deterministic per-instance
```

A counter that lives in process memory is fine for de-duplicating
within one process. It is NOT a safe component of a UNIQUE key
that's persisted across processes. Any process restart resets it
to zero and starts replaying values that already exist in the DB.

Generalizes: **process-local monotonic counters are unsafe as
components of cross-process DB UNIQUE keys without a per-process
salt.** This bug class shows up any time you generate IDs from
local state and persist them under a uniqueness constraint
shared with other processes (or with future versions of the same
process).

# Resolution

Prefix a per-instance random salt onto the counter:

```python
import secrets

class KBRetrievalProcessor(FrameProcessor):
    def __init__(self, ...):
        self._instance_salt = secrets.token_hex(2)  # 4 hex chars = 16 bits
        self._turn_counter = 0

    async def process_frame(self, frame, direction):
        self._turn_counter += 1
        turn_id = f"{self._instance_salt}-{self._turn_counter:05d}"
        # → "a3f7-00001", "a3f7-00002", ...
```

A new processor instance gets a different salt with overwhelming
probability (16 bits = 1/65,536 collision per restart pair).
Counter monotonicity is preserved within a single processor's
lifetime — operators reading "turn N of session S" in logs still
get readable, ordered values.

The fix is one line of state and one line of formatting. No
database changes. No infrastructure changes.

# Why this resolution over alternatives

We considered four options:

1. **UUID per turn** (`turn_id = uuid.uuid4()`). Simplest,
   guaranteed-unique. Rejected because it loses the "turn N of
   session S" readability that on-call relies on when correlating
   logs across services.

2. **Persist the counter** (read `MAX(turn_id)` from DB at processor
   init). Adds a DB hit per restart and depends on the DB being
   reachable when the agent boots — a reachability problem that
   doesn't exist with the salt approach.

3. **Per-instance salt** (chosen). Preserves monotonicity within
   one process; eliminates cross-process collision; one-line fix.

4. **UPSERT-on-conflict** (`ON CONFLICT (session_id, turn_id)
   DO NOTHING`). Loses forensic data — the second turn at the
   same `turn_id` would silently no-op the trace write. Bad for
   the FR-021 forensics contract.

# Detection rule

Any time you have a counter that's BOTH (a) initialized in
`__init__` and (b) part of a DB UNIQUE/PRIMARY KEY constraint,
ask: "What happens to the counter on restart?" If the answer is
"it goes back to zero," you have this bug. The fix is always to
salt with random per-instance state.

The general rule is: **a UNIQUE key whose lifetime is durable
(database) cannot be assembled from components whose lifetime is
ephemeral (process memory) without a per-process disambiguator.**

# Test pinning

```python
def test_turn_id_salt_differs_across_processor_instances():
    """Two processors built in the same session MUST get distinct
    salts so one's counter can never alias onto the other's."""
    p1 = KBRetrievalProcessor(...)
    p2 = KBRetrievalProcessor(...)
    assert p1._instance_salt != p2._instance_salt
```

Plus a regex assertion on the format:

```python
assert re.match(r"^[0-9a-f]{4}-\d{5,}$", body["turn_id"])
```

# References

- Issue #48 (Closed by PR #52)
- Constitution Principle II (test-first, every-writable-column on writes)
- `services/agent/kb_retrieval.py:93-103` (the salted counter)
- `apps/api/migrations/versions/0004_kb_chunks.py` (the UNIQUE constraint that bites)

---
title: "RETRIEVER_INTERNAL_TOKEN rotation playbook with dual-token grace window"
module: services/retriever
date: 2026-04-27
category: docs/solutions/security-issues
problem_type: operational_runbook
component: shared_secret
severity: medium
root_cause: lru_cached_settings_freeze_secret_at_first_load
resolution_type: runbook
symptoms:
  - Rotating RETRIEVER_INTERNAL_TOKEN in infra/.env without restarting both agent + retriever leaves the OLD token authenticating until process restart
  - Uncoordinated rotation produces 401 cascades from the agent → every /retrieve becomes a refusal until the lagging side restarts
  - "kb_retrieval.auth_failed" log lines fire per turn but no synthetic alert pages on-call
tags:
  - secret-rotation
  - bearer-auth
  - operational-runbook
  - dual-token
  - x-internal-token
---

# What this is

The procedure for rotating `RETRIEVER_INTERNAL_TOKEN` (the shared
secret between agent and retriever, header `X-Internal-Token`)
without a downtime window. Use when:

- A leak is suspected and the secret must be rotated NOW.
- A scheduled key-rotation cycle hits its quarterly cadence.
- An engineer with access has left the team.

# The constraint

`@lru_cache(maxsize=1)` on `services/retriever/config.py::get_settings`
means the retriever loads the env once at first request and caches
it forever. Rotating the env without process restart leaves the
old value authenticating. The agent has the same constraint at the
`httpx.AsyncClient` layer — headers are baked into the client at
construction and don't pick up env changes.

So a naive rotation ("update infra/.env, run `docker compose up -d`")
restarts the retriever before the agent and produces a 401-cascade
window where every voice turn becomes a refusal.

The dual-token grace window (`RETRIEVER_INTERNAL_TOKEN_PREVIOUS`)
removes that window.

# The procedure

Three phases. Each phase is a separate compose-up.

## Phase 1 — Stage the new secret as `_PREVIOUS`

In `infra/.env`:

```bash
RETRIEVER_INTERNAL_TOKEN=<old-secret>          # unchanged
RETRIEVER_INTERNAL_TOKEN_PREVIOUS=<NEW-secret> # add the new one here first
```

```bash
docker compose -f infra/docker-compose.yml up -d retriever
```

The retriever now accepts BOTH the old and new tokens. Agents are
still presenting the old one — every turn keeps working.

> **Why stage the new value as `_PREVIOUS` first?** Because at this
> point the agent only knows the old value. We need the retriever
> to accept either old (so today's agents work) or new (so phase 2
> agents can boot ahead of phase 3). Putting the new value into
> `_PREVIOUS` makes both work without needing the agent to know
> the new value yet.

## Phase 2 — Promote the new secret to current; agents pick it up

In `infra/.env`:

```bash
RETRIEVER_INTERNAL_TOKEN=<NEW-secret>            # was old; now new
RETRIEVER_INTERNAL_TOKEN_PREVIOUS=<old-secret>   # was new; now old
```

```bash
docker compose -f infra/docker-compose.yml up -d retriever agent
```

Now:
- Retriever accepts the new (current) AND the old (previous) tokens.
- Agent restarts with the new token in its env, rebuilds its
  `httpx.AsyncClient` with the new header, and starts presenting
  the new token on every turn.

Even if the agent restarts AFTER the retriever in this step, the
retriever still has the old token in `_PREVIOUS` and accepts it
during the brief overlap.

## Phase 3 — Drop the old secret

After confirming all agents are presenting the new token (check
logs for `kb_retrieval.auth_failed` entries — should be zero for
≥10 minutes):

In `infra/.env`:

```bash
RETRIEVER_INTERNAL_TOKEN=<NEW-secret>            # unchanged
RETRIEVER_INTERNAL_TOKEN_PREVIOUS=               # cleared
```

```bash
docker compose -f infra/docker-compose.yml up -d retriever
```

Old secret stops working. Rotation complete.

# Verification at each phase

Smoke test from inside the docker network:

```bash
TOKEN=$(grep '^RETRIEVER_INTERNAL_TOKEN=' infra/.env | cut -d= -f2-)
docker compose -f infra/docker-compose.yml exec agent \
  curl -fsS -H "X-Internal-Token: $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"tenant_id":"00000000-0000-0000-0000-000000000001","session_id":"00000000-0000-0000-0000-000000000002","turn_id":"smoke-1","transcript":"проверка"}' \
  http://retriever:8002/retrieve
```

Expect 200 (or a typed error envelope — anything but 401 / 503
`retriever_unconfigured`).

# Why dual-token over alternatives

| Option | Status |
|---|---|
| **Dual-token grace window** (chosen) | Zero-downtime rotation; one extra env var; ~10 lines of code. |
| **Drop the lru_cache + read os.environ per request** | Microsecond cost per request, but the agent side STILL needs to rebuild its httpx.AsyncClient — doesn't fully solve the rotation problem. |
| **SIGHUP-driven reload** | Adds signal-handling complexity; doesn't help across container restart vs. live-reload semantic mismatch. |
| **Coordinated downtime** | Fully works but requires a maintenance window; users hear refusals during the rollout. |

# Alarm hookup (optional but recommended)

Add a check that pages on-call when `kb_retrieval.auth_failed`
appears in agent logs at any non-zero rate for >5 minutes. The fix
is always "restart the lagging side"; an automated alert lets the
operator catch a missed rotation step before the user does.

# References

- Issue #55 (Closed by PR addressing all open review feedback)
- `services/retriever/auth.py::require_internal_token` (the dual-token check)
- `services/retriever/config.py::Settings.retriever_internal_token_previous`
- `infra/.env.example` (the documented slot)
- Companion runbook: `docs/solutions/security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md` (the original isolated-namespace pattern)

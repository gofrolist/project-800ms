# Voice Stack A/B Experiment — STT Isolation (Whisper vs GigaAM-v3)

**Status:** Draft
**Created:** 2026-04-19
**Horizon:** ~3 weeks total — ~3 days for a WER pre-gate (see below), then up to 2 weeks subjective A/B or until n ≥ 30 rated sessions/stack, whichever hits first.
**Artifact type:** Requirements doc → `ce-plan` next

## Problem

Our current STT is Faster-Whisper (`large` / `int8_float16`). Published 2025–2026 benchmarks claim **GigaAM-v3** (Sber, MIT-licensed, released Nov 2025) cuts Russian WER by ~50% vs Whisper-large-v3 in blind A/B, and is faster on L4. Before committing to an irreversible swap, we want to validate this on real session audio and keep the legacy path running side-by-side for direct comparison.

## Goals

1. Decide within ~3 weeks whether GigaAM-v3 replaces Faster-Whisper as the default STT for Russian voice sessions, using a **hybrid rigorous/directional design** — objective WER pre-gate, then blinded subjective A/B only if WER is inconclusive.
2. Add the smallest surface that still supports a blinded comparison: a stack toggle on the demo SPA with neutral labels, one field on the session API, a per-stack STT-latency histogram.
3. Keep the decision **reversible if WER and rating signals disagree** — winner becomes default, loser kept one release, no delete migration until the signal is unambiguous.
4. Leave a durable evidence trail (WER scores, transcripts, star ratings — all tagged by stack) so the decision is auditable.

## Non-goals

- **No LLM swap** — both stacks use Groq Llama-3.3-70b as today.
- **No TTS swap** — both stacks use Piper (`ru_RU-denis-medium`) as today. F5-TTS / T-Lite are deferred to separate experiments after this one concludes.
- **Not a tenant-facing feature** — this is a demo-UI toggle only. `tenants` table gains nothing. Admin API unchanged.
- **Not a full benchmark harness** — ideation idea #6 (reusable WER/BLEU/TTFT eval suite run on every STT/LLM/TTS change) is still a separate future commit. **This experiment does include a minimal one-shot WER smoke test** as a pre-gate (see "Pre-experiment gate" below); the smoke is not productionised as a reusable tool.
- **No per-component toggles beyond STT** — LLM and TTS are pinned; the `stack` field only switches STT.

## Users

- Primary: solo eval (you).
- Secondary: 3–5 invited playtesters hitting `https://coastalai.ai`.
- Volume: < 50 sessions/day during the experiment window.

## Pre-experiment gate: WER smoke test

Before any SPA work or dispatch-path changes:

- Assemble a fixed Russian eval set of ~100 utterances — mix of clean speech, noisy speech, game-specific vocabulary (character names, quest keywords, place names), stress-mark sensitive words. Hand-label truth transcripts once.
- Script a one-shot harness (`services/agent/scripts/wer_smoke.py`) that feeds each utterance through both `FilteredWhisperSTTService` and the new `GigaAMSTTService` offline, computes WER per stack, prints a table.
- Run once; commit the eval set + truth transcripts + latest WER numbers to the repo.

**Gate outcomes:**

- **Decisive GigaAM win (Δ WER ≥ 5 points, no catastrophic regression on any utterance class):** shortcut — adopt GigaAM without subjective A/B. Skip to "adopt as default, keep legacy one release" cleanup. Saves the 2-week week on the SPA.
- **Decisive Whisper win (GigaAM ≥ 5 points worse):** stop. Document findings, close the experiment, delete GigaAM work. No SPA change ships.
- **Inconclusive (Δ WER < 5 points in either direction, or mixed per class):** proceed with the blinded subjective A/B below.

This gate has the highest signal-to-cost ratio in the whole experiment — if GigaAM is as much better as Sber's published numbers claim, WER alone is enough evidence.

## Decision criterion (subjective A/B, only runs if WER gate is inconclusive)

Signals in priority order (higher beats lower on ties):

1. **WER** from the pre-gate (absolute anchor; always available).
2. **Aggregate star ratings per stack** — mean + min/max, computed only from sessions with ≥ 30 rated (non-skipped) entries per stack.
3. **TTFT per stack** — new `stt_duration_seconds{stack, phase}` histogram in the agent (not the API-level `http_request_duration_seconds`, which times a different layer).
4. **Manual transcript spot-check** — ~10 longest transcripts per stack — used as a tie-breaker when WER and rating disagree.

**Sample size floor:** the subjective decision is only honored when **≥ 30 rated sessions per stack** exist. Below that, the experiment auto-extends regardless of rating-spread.

**Decision matrix (WER × rating):**

| WER gate | Rating | Outcome |
|---|---|---|
| Shortcut: decisive on WER | (not run) | Adopt GigaAM default, keep legacy one release |
| Inconclusive | GigaAM wins rating at n ≥ 30 | Adopt GigaAM default, **keep legacy one release** (conditional cleanup — reversible because signal is only one-dimensional) |
| Inconclusive | Whisper wins rating at n ≥ 30 | Keep Whisper, delete GigaAM branch + new dep |
| Inconclusive | Rating within 0.5-star spread at n ≥ 30 | Tie-breaker: transcript spot-check + TTFT. If neither decisively wins, declare "inconclusive + low impact," keep Whisper as default, document GigaAM as a known alternative |
| Inconclusive | n < 30 per stack after 2 weeks | Auto-extend; if still < 30 after 3 weeks, declare "no decision possible; keep Whisper" |

## Behavior

### Demo SPA (`apps/web`)

**Blinded A/B design** — the rater does not know which stack is which until after submission.

- Idle screen replaces the single "Start call" button with two buttons labeled neutrally: **"Start call (Stack A)"** and **"Start call (Stack B)"**. The mapping (A ↔ legacy, B ↔ gigaam) is randomized **per rater session** (stored in `localStorage` under `stack_ab_mapping` so within-rater comparisons stay consistent across reloads, but different across a browser-data clear).
- In-call `CallView`: **no stack badge**. The only visible context is the neutral "Stack A" or "Stack B" label on the active call. No way to swap mid-call.
- On End call: modal titled "Rate this call" blocks the idle transition. 1–5 star picker + optional comment textarea (max 2000 chars). After submit (or Skip), the modal **reveals the mapping for this call only**: e.g., "You just rated Stack B — this was GigaAM-v3." Reveal is informational; doesn't alter the submitted rating.
- "Submit" writes to DB; "Skip" dismisses without writing any row to `session_ratings` — `GET /v1/sessions/{room}.rating` remains `null` for skipped sessions. Fresh sessions start from idle after either.
- Error cases: if the session POST fails for the new stack, don't silently fall back to legacy — show the error envelope's `error.message`. The in-UI error shows the neutral label ("Stack B failed — retry, or try Stack A") to preserve the blind.

### API (`apps/api`)

- `CreateSessionRequest` gains an optional field:
  ```
  stack: Literal["legacy", "gigaam"] | None = None
  ```
  Default on the server side: `"legacy"`. Persists to `sessions.stack`.
- Dispatch payload to the agent includes `stack` alongside the existing `persona`/`voice`/`language`/`llm_model`/`context`.
- New endpoint:
  ```
  POST /v1/sessions/{room}/rating
  body: { rating: 1..5, comment?: string (max 2000 chars) }
  ```
  Requires `X-API-Key`; depends on `enforce_tenant_rate_limit` (matching every other `/v1/*` route) so the 404/409 surface can't be used as a brute-force enumeration oracle.
  **Check ordering (to avoid a cross-tenant existence leak):** the handler first joins on `sessions.tenant_id = identity.tenant_id` and 404s if the session doesn't exist *or* belongs to another tenant (same no-leak contract as `GET /v1/sessions/{room}`). Only after that does it check the unique index on `session_ratings.session_id` and return 409 if a rating already exists (idempotent — don't create two). Returns 201 on success.
  The `comment` field is stored raw (no markdown / HTML rendering); any future admin UI must HTML-escape on render.
- `GET /v1/sessions/{room}` response shape gains:
  - `stack: "legacy" | "gigaam" | null`
  - `rating: int (1..5) | null` (null when no rating submitted)
  - `comment` is **not** exposed on this endpoint — surfaced only via a future admin-scoped endpoint (out of scope for this experiment).
- `GET /v1/usage` is unchanged. A separate admin-only summary endpoint ("ratings by stack") is out of scope for this experiment.

### Agent (`services/agent`)

**Dependencies:**

- `gigaam` via `git+https://github.com/salute-developers/GigaAM@<pinned-sha>` in `services/agent/pyproject.toml` (PyPI 0.1.0 is v1/v2-only; v3 requires git source). Pin the sha; don't follow `main`.
- PyTorch + matching CUDA wheels are a new runtime dependency alongside ctranslate2. Must be ABI-compatible with the existing CUDA 12.x / cuDNN 9 base image. Image size expected to grow ~2–3 GB; acceptable for experiment.
- GigaAM-v3 weights (`ai-sage/GigaAM-v3` on HF) are **baked into the Docker image at build time**, not downloaded at first-run. Eliminates cold-start latency as a confound.

**Model loading:**

- Both Whisper and GigaAM models are **preloaded at agent startup**, not lazily. The current `load_whisper()` idiom in `services/agent/main.py` gains a sibling `load_gigaam()`. Both live in RAM for the lifetime of the process.
- **VRAM verification gate (day-1):** before writing the Pipecat wrapper, run `nvidia-smi` on the dev L4 with Whisper-large + GigaAM-v3 + Silero VAD + vLLM (Qwen-7B-AWQ) all resident. Fail the milestone if steady-state usage > 22 GB on a 24 GB L4.
- **Startup error handling:** if `load_gigaam()` fails, the agent **fails startup entirely** (exit non-zero so the compose health check catches it). We explicitly do not mark-unavailable-and-continue — silent degradation of the new stack would contaminate the experiment.

**Dispatch + stack routing:**

- `PerSessionOverrides` gains `stack: str | None` parsed from the dispatch payload. The agent **independently whitelists** the value against `{"legacy", "gigaam"}` (trust boundary — API validation is not sufficient on its own) and falls back to `"legacy"` on any unknown value with a warning log.
- `pipeline.build_task` branches on `overrides.stack`: `"gigaam"` wires the new `GigaAMSTTService`, anything else uses the existing `FilteredWhisperSTTService`.
- **Kill switch:** the API container reads an env var `DISABLE_STACK_GIGAAM` (default empty). When set (any truthy value), any `CreateSessionRequest.stack="gigaam"` is silently coerced to `"legacy"` with a warning log in the API. This is the in-experiment rollback lever — no agent restart needed, both models stay loaded, just stop routing traffic to the broken path. No SPA change required.

**GigaAMSTTService:**

- Subclasses `pipecat.services.stt_service.SegmentedSTTService` (not `WhisperSTTService` — GigaAM has no Whisper-style segment stats, and per-VAD-segment matches the current pipeline wiring). First cut: utterance-level transcription.
- **Filter parity (pinned as a second variable):** the new hallucination filter uses the equivalent of the Whisper filter's *behaviour*, not its *signals*. Concretely: same reject criteria — utterance < 300ms → drop (matches VAD short-segment behaviour), utterance with only 1 token → drop, otherwise accept. We measure and tune these thresholds on the WER pre-gate eval set so both stacks reject roughly the same proportion of false positives on the same ground-truth audio. Documented as a "pinned variable" — if the filter is what moves the rating, we want to know.

**Metrics:**

- New Prometheus histogram in the agent: `stt_duration_seconds{stack, phase}` where `phase` ∈ `{decode, filter, total}`. Emitted from the STT service wrapper itself so we measure the actual STT layer, not the upstream API request. This replaces the draft's reference to `http_request_duration_seconds` which times a different layer.

### Data model

New Alembic migration **0004_stack_and_ratings**:

- Add `stack TEXT NOT NULL DEFAULT 'legacy'` to `sessions`. Backfill existing rows to `'legacy'` in the same migration (`UPDATE sessions SET stack = 'legacy' WHERE stack IS NULL;` then `ALTER COLUMN ... SET NOT NULL`). No CHECK constraint — we might extend to more stacks later. The "NULL means legacy" contract is now schema-enforced, not prose.
- New table `session_ratings`:
  - `id UUID PRIMARY KEY` (server-generated via pgcrypto, pattern from existing tables)
  - `session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE`
  - `rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5)`
  - `comment TEXT NULL` (enforced at the Pydantic layer to ≤ 2000 chars)
  - `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
  - Unique index on `session_id` (one rating per session).
- **Tenant isolation pattern (matches `session_transcripts` precedent):** `session_ratings` has no direct `tenant_id` column — it inherits tenant ownership via `session_id → sessions.tenant_id`. Every read/write must scope through a JOIN on `sessions.tenant_id = caller.tenant_id`. A naive `SELECT ... WHERE session_id = ?` without the join is a cross-tenant leak.

## Success criteria (shippable checklist)

**Phase A — WER pre-gate:**
- [ ] `services/agent/scripts/wer_smoke.py` runs both stacks offline against a committed ~100-utterance Russian eval set.
- [ ] WER numbers + latest run date committed to the repo.
- [ ] VRAM verification: `nvidia-smi` shows < 22 GB steady-state with both STT models + vLLM co-loaded.
- [ ] Filter parity pinned: both stacks reject roughly the same proportion of false-positive utterances on the eval set (target within 10% of each other).

**Phase B — SPA A/B (only runs if Phase A is inconclusive):**
- [ ] Two neutral-labelled buttons ("Stack A" / "Stack B") render in the SPA; mapping randomized per rater in `localStorage`.
- [ ] No in-call stack badge; mapping revealed post-submit only.
- [ ] `sessions.stack` persists the chosen value for every session (NOT NULL default handled).
- [ ] GigaAM audio flows end-to-end: POST → dispatch → agent → LiveKit → response audio in browser.
- [ ] Rating modal submits; writes exactly one row per session to `session_ratings`; post-submit mapping reveal renders.
- [ ] `GET /v1/sessions/{room}` returns both `stack` and `rating` (comment excluded).
- [ ] Zero regressions on sessions that don't pass the new field.
- [ ] Rating endpoint tenant-scoped — cross-tenant rating attempts 404.
- [ ] `stt_duration_seconds{stack, phase}` histogram emitting from the agent.
- [ ] `DISABLE_STACK_GIGAAM=1` kill switch verified in staging: setting it silently reroutes `stack="gigaam"` requests to legacy.

## Known unknowns & implementation risks

Addressed; repeated here as a single glance-at checklist for planning.

1. **GigaAM Pipecat integration** — no published service class. Mitigation: subclass `SegmentedSTTService`, utterance-level first cut; the WER pre-gate lets us fail fast if the wrapper has quality regressions independent of the model.
2. **VRAM fit** — day-1 `nvidia-smi` measurement gate with 22 GB ceiling. Agent fails startup hard if either model doesn't load.
3. **Hallucination filter parity** — pinned as an explicit second variable; same reject semantics on both, thresholds tuned on the eval set.
4. **Rating bias** — blinded A/B via neutral labels + post-submit reveal.
5. **Latency confounding** — `stt_duration_seconds{stack, phase}` histogram in the agent layer.

## Cleanup plan (conditional, reversible)

Driven by the WER × rating decision matrix above. The experiment has an end date + an owner-assigned follow-up ticket; cleanup is not a suggestion.

**Scenario 1 — Decisive agreement (WER shortcut OR WER-inconclusive + clear rating winner):**
- Winner becomes the default (API's server-side default flips to the winner's `stack` value, SPA single-button with "Start call" returns).
- **Loser is kept for one release** as a safety net. The `stack` field stays on the API (now documented as "switch to legacy if needed"); the loser's Pipecat service + model preload stays in the agent.
- Follow-up ticket (owner: you, due: 4 weeks after decision): delete loser, drop `sessions.stack` column + `session_ratings` table in migration 0005, remove the kill-switch env var.

**Scenario 2 — Disagreement (WER and rating point different directions, no decisive winner):**
- Default does not change — Whisper stays. GigaAM documented as "directionally interesting, not adopted; try again after the voice model landscape shifts."
- Delete GigaAM code + `gigaam` dep + PyTorch + baked weights. Drop `sessions.stack` + `session_ratings` in migration 0005 immediately.

**Scenario 3 — No decision possible (n < 30/stack after 3 full weeks):**
- Experiment ends. Default stays Whisper. Treat as Scenario 2's cleanup (delete GigaAM path) — we do **not** leave the toggle running indefinitely. The "durable dev toggle" option was rejected at brainstorm; re-committing to the experiment until volume arrives is the forcing function.

**Forcing function:** the cleanup follow-up ticket is opened on the same day the decision is made. Calendar reminder at +4 weeks. If no decision within 3 weeks, cleanup runs the next calendar week regardless.


## Out of scope (deferred deliberately)

- F5-TTS / T-Lite swaps — separate brainstorm + experiment per component.
- Automated objective WER harness (ideation idea #6) — complementary but separate work.
- Admin UI for experiment summary — if needed, a SQL query gets us there during the 1-week window.
- Tenant-level "preferred stack" setting — explicit non-goal, we're not productizing the toggle.

## Doc-review resolution summary

Doc reviewed by 6-persona panel on 2026-04-19. P0s and top P1s resolved via **Option C (Hybrid — WER pre-gate + blinded subjective + conditional reversible cleanup)**. Resolutions are woven throughout the doc above:

- **Rating bias** → blinded "Stack A / B" labels, post-submit reveal only, no in-call badge.
- **Decision weakness** → conditional cleanup (reversible when signals disagree), n ≥ 30/stack floor.
- **Dependency story** → `git+https` pinned sha + weights baked into image.
- **WER-first gate** → adopted as Phase A; short-circuits the SPA work if decisive.
- **VRAM envelope** → day-1 measurement gate (22 GB ceiling on 24 GB L4).
- **`stack` NOT NULL** → schema-enforced with backfill.
- **Filter parity** → pinned as explicit second variable (same reject semantics, tuned on the eval set).
- **Mixed-outcome matrix** → explicit table in the Decision criterion section.
- **Skip-rate threshold** → 30 rated sessions/stack floor.
- **TTFT layer** → new `stt_duration_seconds{stack, phase}` histogram in the agent.
- **Partial rollback** → `DISABLE_STACK_GIGAAM` env var on the API; no agent restart needed.
- **Forcing function** → cleanup ticket opened on decision day, runs at +4 weeks regardless.

### Still-open P2s (resolve in planning, not here)

1. **Rating modal UX states** (design-lens). Dismiss behavior (Esc / backdrop / tab close / LiveKit disconnect) + submit states (loading / error / 409 / timeout) unspecified. Plan should spec these.
2. **A11y for new surfaces** (design-lens). Focus trap, role=dialog, keyboard star picker (radio semantics, not divs), mic-permission-denied flow.
3. **Minimum session length** (adversarial). Greeting + preloaded TTS is STT-independent; short calls dilute signal. Consider: reject sessions < 20s of caller speech from the rating aggregate.
4. **Retention for `session_ratings.comment`** (security). Free-text Russian user input with no retention policy. Plan: follow the same retention policy as `session_transcripts` (inherited via session_id), documented once the broader retention story exists.
5. **Correcting a misrating** (adversarial). 409-on-duplicate means a misclicked 5-instead-of-2 is permanent. With n~30, one misclick is ~3% of the dataset. Plan: either expose `PATCH /v1/sessions/{room}/rating` (same-tenant-owner overwrite) or change the unique index to "latest rating wins" with history retained.

### Residual risks (noted, accepted for the experiment)

- "GigaAM faster on L4" is Sber-published; our own latency numbers from `stt_duration_seconds` will settle it during the WER gate.
- Cross-session rater fatigue: solo rater doing many back-to-back sessions may have rating drift confounded with stack order. Mitigation: the blinded randomization is per-rater-session, not per-call — both stacks see similar order distributions over time.
- `coastalai.ai` is production traffic; strangers hitting the SPA will see two "Stack A/B" buttons with no explanation. Consider brief intro copy or gating the experiment behind an explicit dev path (`/experiment` route). Low-impact but worth a line in planning.

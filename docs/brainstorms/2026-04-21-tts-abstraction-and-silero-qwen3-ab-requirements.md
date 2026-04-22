---
date: 2026-04-21
topic: tts-abstraction-piper-silero-qwen3-ab
---

# TTS Provider Abstraction + Piper / Silero v5 / Qwen3-TTS A/B Experiment

**Status:** Draft — refined 2026-04-21 after ce-doc-review (round 1)
**Created:** 2026-04-21
**Horizon:** ~3 weeks total — ~1 day Phase 0 (blind premise check + wrapper/adapter smokes), ~3 days objective pre-gate, then up to 2 weeks subjective phase (mode TBD — see Outstanding Questions). Abort branches collapse the horizon earlier.
**Artifact type:** Requirements doc → `ce-plan` next

## Problem Frame

Current agent TTS is Piper (`ru_RU-denis-medium`, CPU, MIT, wired at `services/agent/pipeline.py:117-121`). Team judgment is that the Russian voice sounds robotic, which hurts perceived quality even as STT (GigaAM-v3 since the prior A/B) and LLM responses keep improving. No objective data exists to anchor the complaint or rank replacements.

The ideation round (2026-04-21) narrowed self-hosted Russian-capable, MIT/Apache-licensed, L4-fitting candidates to three — Piper stays in the comparison so the "is the premise real?" question is answerable:

- **Piper** *(incumbent)* — current production engine, CPU, MIT, already in `pipeline.py`. Treated here as a **first-class adoption candidate**, not just a baseline reference: if Piper's objective + subjective scores are competitive, staying on it is a legitimate outcome.
- **Silero v5** — Russian-native, UTMOS 3.04 on the `v5_cis_base` model, MIT, ~200 MB weights, RTF 0.06 on GPU (~30-60 ms). **No Pipecat TTS adapter exists** — we write one.
- **Qwen3-TTS** — Apache-2.0, 10 languages incl. Russian, ~97 ms TTFB claimed, 0.6 B and 1.7 B variants. **No Pipecat adapter upstream**, but a community OpenAI-compatible FastAPI wrapper (`dingausmwald/Qwen3-TTS-Openai-Fastapi`) lets us plug into Pipecat's existing `OpenAITTSService` without new adapter code on the agent side.

The prior Whisper→GigaAM STT experiment (`docs/brainstorms/2026-04-19-voice-stack-ab-experiment-requirements.md`) established the A/B pattern we reuse: objective pre-gate → conditional subjective blinded A/B. That doc explicitly deferred TTS-A/B infrastructure to a future commit — this brainstorm is that commit.

## Requirements

**Phase 0 — premise check and risk spikes** (gates all downstream work)
- **R21.** Blind premise-listen gate. Before any code is written, ≥ 2 team members listen blind to 10 identical Russian utterances rendered by Piper (current production output), Silero v5 (published demo), and Qwen3-TTS (published demo). Aggregate rank per listener; if Piper is not ranked consistently last by ≥ 50% of listeners, abort the experiment and file a "Piper is acceptable for MVP" solution doc. Cost: ~1 hour. Falsifies the unvalidated "Piper sounds robotic" premise before the 3-week commitment.
- **R22.** Silero adapter spike. A 4-hour timeboxed investigation before R4 commits to the 3-day horizon. Outputs a single-page summary covering: (a) Silero v5 emit sample rate vs LiveKit transport expectation + resampler path; (b) Pipecat `TTSService` frame cadence contract and whether file-at-a-time synth is actually compatible with the transport's audio scheduler; (c) barge-in / interruption contract — how a file-at-a-time engine behaves when `InterruptionFrame` arrives mid-synth; (d) GPU placement / async model interaction. If any of these reveal a week-class issue, the Silero track is re-scoped before the horizon is committed.
- **R23.** Qwen3 wrapper smoke. A 10-minute validation before R8 commits: install `dingausmwald/Qwen3-TTS-Openai-Fastapi` pinned to a specific commit SHA, confirm the wrapper responds to `/v1/audio/speech` requests, emits audio at the sample rate it advertises, handles streaming vs file-at-a-time correctly for Pipecat's `OpenAITTSService`. Also confirm the wrapper's license is compatible with vendoring into our repo. **If the wrapper is broken, unlicensed, or non-compat, drop the Qwen3 track — do not fork-and-maintain mid-experiment.**

**TTS Provider Abstraction**
- **R1.** Introduce a `TTSProvider` selection seam inside `services/agent/` that resolves to a Pipecat `TTSService` instance based on config. Day-one values: `piper` (existing), `silero`, or `qwen3`.
- **R2.** `pipeline.py` references the abstraction, not a specific TTS class. Swapping a provider is a config change, never a code change.
- **R3.** The abstraction preserves the existing `PerSessionOverrides.voice` contract — a caller can request a specific voice per session without knowing which engine renders it. Providers decide how to map voice IDs (e.g., Silero speaker name, Qwen3 voice ID, Piper voice file).

**Silero v5 integration**
- **R4.** Ship a `SileroTTSService` subclass of Pipecat's `TTSService` in `services/agent/silero_tts.py`. Russian-only for this experiment; the class must expose `language`/`speaker` kwargs but may reject non-Russian values.
- **R5.** Model weights cache into the existing `hf_cache_agent` named volume (same pattern as GigaAM). No per-model ad-hoc volume.
- **R6.** Preload the model at agent startup (mirror `load_gigaam()` in `services/agent/models.py`) so the first session doesn't pay a cold load.
- **R7.** File-at-a-time synth is acceptable for this experiment — do not require streaming from Silero. (Pair with the upstream `LLMFullResponse` buffer fix separately; that's ideation survivor #5, not in scope here.)

**Qwen3-TTS integration (sidecar)**
- **R8.** Add a `qwen3-tts` service to `infra/docker-compose.yml` using the community OpenAI-compatible FastAPI wrapper. Runs on the same L4 GPU as the agent. Bind admin port to `127.0.0.1` (same pattern as `vllm`). Put it behind a `tts-qwen3` compose profile so it only starts when the Qwen3 track is active. **Deploy the 0.6 B variant as the default for this experiment** — VRAM co-residency math (see Dependencies) puts the 1.7 B variant at the edge of the L4 ceiling alongside vLLM-7B. 1.7 B is evaluated only if 0.6 B underperforms on the pre-gate AND a co-resident VRAM measurement confirms it fits under production load.
- **R8a.** Pin the community wrapper to a specific commit SHA in `infra/docker-compose.yml` (no `:latest`, no branch name). Vendor a known-good copy into the repo under a path we can fork/patch (e.g., `infra/qwen3-tts-wrapper/`) so an upstream delete/force-push does not brick our build. Confirm the wrapper's license is compatible with our repo (MIT/Apache-compatible) before R8 lands — record the license in `docs/solutions/tts-selection/wrapper-license.md`.
- **R8b.** Dedicated HF cache volume for the Qwen3 sidecar: `hf_cache_qwen3` (NOT `hf_cache_agent`). The existing agent↔vllm cache split exists in compose precisely because different containers run under different UIDs and share-mounting breaks write permissions (documented inline in `infra/docker-compose.yml`). The sidecar gets its own volume on the same pattern.
- **R9.** The agent talks to the sidecar over the internal docker network via Pipecat's existing `OpenAITTSService` — no new Pipecat adapter code on the agent side.
- **R10.** Qwen3 model variant (0.6 B vs 1.7 B) is a sidecar config, not an agent config — the agent only knows the OpenAI-compatible endpoint.

**A/B evaluation harness**
- **R11.** Build a one-shot harness (`services/agent/scripts/tts_bench.py`) that, given a fixed Russian eval set, runs **all three adoption candidates (`piper`, `silero`, `qwen3`)** through the R1 abstraction and emits a scorecard: UTMOS (mean + per-class), TTFB p50/p95, RTF, VRAM-at-synth, output sample rate. All three engines are first-class candidates — Piper staying in production is a valid outcome.
- **R11a.** Harness VRAM measurement runs with **all sibling services co-resident** (`docker compose --profile local-llm,tts-qwen3 up` or equivalent), not standalone. VRAM measured on a cold agent is misleading — the production failure mode is OOM when vLLM-7B + GigaAM + TTS all hold GPU state simultaneously.
- **R11b.** Harness also measures current-Piper **end-to-end first-audio TTFB p95** on a live session replay (or equivalent in-pipeline timing) so the latency-regression gate in Success Criteria has a real baseline number. Target file: the same scorecard emitted by R14.
- **R12.** Eval set is a committed ~50-utterance Russian corpus — mix of short conversational (~5 words), medium (~15 words), long (~40+ words), and utterances with brand/persona/game-specific vocabulary. Text-only in-repo; no reference audio needed. Each utterance row carries a class label (`short` | `medium` | `long` | `domain`) used by the harness for per-class UTMOS reporting — the per-class UTMOS floor in Success Criteria requires this labeling.
- **R13.** UTMOS scoring uses a pinned model reference so runs are reproducible across laptops and CI. The exact model to pin is a planning-time decision (see Outstanding Questions).
- **R14.** Scorecard output commits under `docs/solutions/tts-selection/` with YAML frontmatter matching the repo's solution format (`module`, `tags`, `problem_type`).

**Offline listening panel (conditional — runs only if the pre-gate is inconclusive)**

> **Design decision (2026-04-21 refinement):** offline-panel mode over in-SPA live-traffic A/B. Rationale: 5 playtesters × <50 sessions/day × 3 engines makes statistically-meaningful per-arm n unreachable in the experiment horizon; the SPA `stack` toggle, rating modal, `/ratings` endpoint, ratings DB table, and agent metrics surface were all deleted in the post-GigaAM cleanup, so R15-R17 as originally framed would be a full re-build for a weakly-powered result. An offline panel with ≥3 blinded raters on a fixed utterance set delivers the same decision signal in 1-2 days with no new SPA/API/DB/metrics scope. In-SPA A/B can be added as a follow-up experiment once real traffic exists.
- **R15.** Offline-panel synth harness. Extend `tts_bench.py` (or add `tts_panel.py` as a sibling script under `services/agent/scripts/`) that, given a ≥30-utterance slice of the R12 corpus, synthesizes each utterance on every surviving candidate (Piper + Silero + Qwen3, minus any dropped by R22 or R23). Writes a blinded on-disk structure: `panel/<date>/blinded/<utterance-id>/<random-label>.wav` with the `random-label → engine-id` mapping stored **outside** the rater-accessible path (e.g., `panel/<date>/_unblind.json`) so raters cannot see which engine produced which clip.
- **R16.** Panel protocol. ≥3 team members rate each blinded clip independently on a 5-point naturalness scale, with an optional free-text failure note. Raters work in isolation — no Slack-channel comparison, no shared doc — until aggregation. Ratings commit as `panel/<date>/ratings-<rater-id>.csv` with columns (`utterance_id`, `label`, `score`, `notes`). The pre-registered analysis plan (`docs/solutions/tts-selection/analysis-plan.md`, committed before the panel runs) fixes the decision rule before data collection.
- **R17.** Aggregation + analysis. Per-engine median score, per-engine interquartile range, Wilcoxon signed-rank test for each engine-pair. Decision threshold anchored to inter-rater variance observed on a 5-utterance pilot run — not a fixed-delta rule — so we don't declare winners on noise-level differences. Final scorecard commits under `docs/solutions/tts-selection/panel-<date>.md` with YAML frontmatter and links to the raw rating CSVs.

**Production selection mechanism**
- **R18.** Provider is an agent env var (`TTS_PROVIDER=piper|silero|qwen3`). Default stays `piper` until the pre-gate is run. The abstraction at R1 reads this value. No per-session override in this experiment (offline-panel mode does not need live-traffic routing); if dev-time per-call switching is needed later, add it when there's a concrete reason.
- **R20.** Keep the **two non-winning providers** deployable for one release after the winner is picked. Reversal must require only a config change, no redeploy of different code. "One release" is defined below in Dependencies.

## Success Criteria

**Phase 0 gate (R21-R23):**
- **R21 premise rejected** (Piper not ranked consistently last) → abort, commit "Piper is acceptable for MVP" solution doc, zero downstream work.
- **R22 spike reveals week-class Silero issue** → re-scope or drop the Silero track; document the constraint.
- **R23 wrapper broken or unlicensed** → drop the Qwen3 track; experiment reduces to Piper-vs-Silero.

**Objective pre-gate (R11-R14), 3-way comparison:**
- **Decisive winner** — one engine's mean UTMOS on the eval set is ≥ 0.3 higher than **both** other engines AND its TTFB p95 is ≤ 500 ms on L4 AND no utterance class drops below 2.5 UTMOS AND its end-to-end first-audio p95 does not regress vs measured Piper baseline by more than 100 ms (the **latency-budget gate**, anchored to R11b's Piper measurement rather than a fixed 500 ms absolute). On a candidate Decisive winner, require a final **blind human sanity-check**: ≥ 2 team members listen to the same 10-sample subset, confirm the UTMOS ranking matches their ear, sign off. Only then flip `TTS_PROVIDER`.
- **Piper wins** — Piper's mean UTMOS is within 0.2 of the best other engine AND Piper's subjective sanity-check is acceptable. Keep Piper as default, commit the scorecard, close the experiment.
- **Inconclusive** — no single engine is ≥ 0.3 above both others, OR two engines tie within 0.2 of each other above Piper. Proceed to the subjective phase (mode TBD in Outstanding Questions).
- **All fail** — every engine's UTMOS falls below Piper on ≥ 2 utterance classes. Abort with findings under `docs/solutions/tts-selection/`; re-ideate (cloud baseline, F5-TTS from-scratch training, etc.).

**Offline listening panel (R15-R17, only if pre-gate is inconclusive):**
- ≥ 3 blinded raters × ≥ 30 utterances × surviving candidates. Per-engine median + IQR. Wilcoxon signed-rank for each engine-pair.
- Decision threshold calibrated on a 5-utterance pilot against observed inter-rater variance — **do not port the Δ ≥ 0.3 star rule from earlier drafts**, it has no statistical grounding here.
- Winner adopted only when (a) the engine-pair test reaches significance under the pre-registered analysis plan AND (b) the Decisive-winner blind sanity-check passes. Analysis plan (`docs/solutions/tts-selection/analysis-plan.md`) committed before the panel runs, to avoid motivated-reasoning in ambiguous results.
- If no engine reaches significance above Piper → Piper wins, scorecard + panel commit, experiment closes.

## Scope Boundaries

- **No STT change** — GigaAM-v3 stays.
- **No LLM change.**
- **No multilingual TTS work.** Silero v5 here is Russian-only. Qwen3-TTS supports others but we do not add English (or any non-Russian voice) in this experiment. R1's abstraction is shaped to allow it later; we do not exercise that path.
- **No voice-cloning.** Zero-shot speaker-reference features (XTTS-style) are out.
- **No SSML / inline prosody steering.** No LLM-emitted style tags.
- **No cloud TTS baseline.** Explicitly rejected in ideation — keeps scope tight. Deepgram Aura-2 or Cartesia can be added in a follow-up experiment if both self-hosted engines disappoint.
- **No tenant-facing change.** `tenants` table unchanged, admin API unchanged, no new session-body fields.
- **No in-SPA A/B toggle or rating UI.** Explicitly deferred — offline-panel mode delivers the decision signal we need at MVP scale; in-SPA A/B is a follow-up experiment once real traffic exists.
- **No agent metrics export surface in this experiment.** Prometheus client, `/metrics` endpoint, histogram registry are out of scope — offline-panel mode does not require them. Added later when live-traffic A/B or production telemetry needs surface.
- **No per-session TTS override API field.** `tts_stack` on `POST /v1/sessions` is out — the offline panel runs against fixed utterance sets, not live sessions, so per-session routing is not required.
- **Not a full reusable LLM/STT/TTS benchmark harness.** The pre-gate harness lives as `services/agent/scripts/tts_bench.py` — a one-shot tool. Promoting it to a cross-component rig is a separate future commit.
- **No delete-and-forget migration.** All three engines remain in the codebase for one release after a winner is picked (R20).
- **No upstream of the Silero Pipecat adapter in this experiment.** Write it in-repo first; upstreaming is a separate follow-up once the API shape is validated in production.

## Key Decisions

- **Three engines, Piper included as first-class candidate** *(refinement 2026-04-21)* — the prior decision was 2 engines with Piper as baseline reference only; updated so "Piper is fine, stay on it" is a recognized valid outcome, not an experimental failure mode.
- **Abstraction-first over pick-first** *(ideation, 2026-04-21)* — seam pays back on every future TTS/STT/LLM swap. Trade-off: the STT experiment did not use an abstraction (inline `GigaAMSTTService` in `pipeline.py`) and it worked fine for a 2-arm experiment; with 3 engines + explicit "keep losers one release" (R20), the registry pattern is load-bearing here in a way it wasn't then.
- **No cloud baseline** *(brainstorm Q2, re-affirmed in refinement)* — keeps scope tight. Cloud is an explicit scope-boundary bet on self-hosted-only (privacy, sovereignty, unit-cost at scale) — the rationale is now named, not just "scope hygiene". Cloud reference can be added as a follow-up if all three engines fail the pre-gate.
- **Qwen3 via sidecar, not in-process** *(brainstorm Q4)* — mirrors the `vllm` service pattern, avoids coupling Qwen3 model loading to agent cold-start. Wrapper is pinned + vendored (R8a) to remove the upstream-abandonment risk.
- **Phase 0 premise check before any code** *(refinement 2026-04-21, R21)* — the "Piper sounds robotic" premise is unvalidated team judgment. A 1-hour blind listen validates or kills the experiment before committing 3 weeks.
- **UTMOS Δ ≥ 0.3 + blind-listen gate** *(refinement 2026-04-21)* — the prior "shortcut-adopt on UTMOS alone" was inconsistent with the own "UTMOS is comparative-only on Russian" disclaimer. Shortcut still exists but now requires a 2-person blind listen on 10 samples before flipping `TTS_PROVIDER`. Cost: 20 minutes. Closes the UTMOS-mis-ranking failure mode.
- **Latency-budget gate anchored to measured Piper baseline** *(refinement 2026-04-21)* — the prior "TTFB p95 ≤ 500 ms" absolute floor consumed 60%+ of the <800 ms end-to-end budget without grounding in current behavior. Now gated on "no regression > 100 ms vs measured Piper baseline" from R11b.
- **Piper stays the default until the pre-gate runs** — no silent-swap risk.
- **Offline listening panel over in-SPA live-traffic A/B** *(refinement 2026-04-21)* — at 5 playtesters + <50 sessions/day + 3 engines, statistically-meaningful per-arm n is unreachable in horizon; SPA+rating+metrics infrastructure was deleted in cleanup so the rebuild cost outweighs the signal. Offline panel (≥3 blinded raters × ≥30 utterances) delivers the same decision signal in 1-2 days with no new SPA/API/DB/metrics scope. Live-traffic A/B re-considered once real traffic exists.

## Dependencies / Assumptions

- **"One release" definition** (for R20). One full Monday-to-Friday iteration of the git `main` branch — concretely, a losing provider's code, weights, and compose service survive in the tree for ≥ 7 calendar days after the winner is flipped in `TTS_PROVIDER`. Deletion PR is opened on day-7, merged on day-10 after a QA pass. Gives a working-week for silent regressions to surface before we lose the rollback.
- **GPU budget.** L4 has 24 GB; current floor is GigaAM (~1 GB) + optional vLLM 7 B (~13 GB) + CUDA/framework overhead (~1 GB) = ~9 GB free. Silero (~200 MB) is noise. Qwen3-TTS 0.6 B fits comfortably; 1.7 B weights alone (~3.4 GB fp16) + AR activation state could push 6-10 GB — at the edge. R11a's co-resident VRAM measurement is the decision rule, not a logging afterthought. Back-of-envelope: 1.7 B + vLLM-7B is a bad bet unless vLLM's `gpu_memory_utilization` is tuned down.
- **Community wrapper viability** (R23 gate). `dingausmwald/Qwen3-TTS-Openai-Fastapi` is the critical path for Qwen3 integration. Pre-gated by R23 smoke + license check + commit-SHA pin + repo vendoring (R8a). If any of those fail, the Qwen3 track is dropped — not forked — and the experiment proceeds Piper-vs-Silero only.
- **UTMOS as a Russian proxy.** The UTMOS22 model was trained primarily on English/Japanese. Russian scores are **comparative-only** — we use deltas between engines, not absolute values against published MOS literature. This is why subjective A/B remains the final arbiter on inconclusive pre-gate.
- **Offline-panel mode eliminates the SPA/API/DB/metrics rebuild** that a live-traffic A/B would have required. R15-R17 are now file-system-scoped harness work, not application infrastructure.
- **`TTS_PROVIDER` env plumbing.** The prior STT A/B's `STT_STACK` env-var + `main.py` per-dispatch threading pattern was removed in cleanup. Re-use the **shape** for `TTS_PROVIDER` — treat as new implementation, not reuse.

## Outstanding Questions

### Resolve Before Planning

*None — product decisions resolved.*

### Deferred to Planning

- [Affects R8][Decided] Qwen3-TTS variant: **0.6 B is the default** (committed in R8). 1.7 B is deferred — evaluated only if 0.6 B underperforms AND R11a co-resident VRAM measurement confirms it fits.
- [Affects R13][Needs research] Which UTMOS model to pin. Default is `sarulab-speech/UTMOS22-strong`. R13 requires a Spearman ρ check against 5-10 human-labeled samples before pinning; record ρ in the scorecard so readers can judge how much weight to give UTMOS on Russian.
- [Affects R11, R11a][Technical] Run harness as a standalone `uv run` script or as a short-lived container? Script first, containerize only on flakiness. R11a requires `docker compose --profile` orchestration so the harness must either invoke compose itself or run inside a compose-aware context.
- [Affects R4, R22][Technical] Silero adapter location — in-repo only, or upstream PR to `pipecat-ai/pipecat`. In-repo first (scoped out above); upstream once the API shape is production-validated. R22 spike outcome determines whether the in-repo adapter is 1-2 days or a week.
- [Affects R11b][Technical] How to measure "current-Piper end-to-end first-audio TTFB p95" baseline — from ad-hoc instrumentation on a fixed replayed session, or from a dedicated harness run? Planning picks; without metrics infra (explicitly out of scope), the measurement is a one-off timing dump, not a rolling percentile.
- [Affects R3, R18][Technical] Semantics of `PerSessionOverrides.voice` across providers. Recommend: `voice` stays provider-specific (each provider interprets the string in its own namespace); `TTS_PROVIDER` selects the interpreter. Document in the provider interface.
- [Affects R15-R17][Product] Pre-register the panel analysis plan (`docs/solutions/tts-selection/analysis-plan.md`) before the offline panel runs — planning writes the first draft, reviewed by whoever will be a rater, finalized before any clip is synthesized.
- [Affects R21][Logistics] Who the ≥ 2 Phase-0 blind-listen raters are. Solo-eval is acceptable for R21 as a tiebreaker only if n=1 is unambiguous (Piper obviously last OR obviously acceptable); otherwise need a second set of ears.

## Next Steps

-> `/ce-plan` for structured implementation planning.

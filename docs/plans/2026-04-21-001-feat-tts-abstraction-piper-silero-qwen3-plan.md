---
title: "feat: TTS Provider Abstraction + Piper/Silero/Qwen3 A/B Experiment"
type: feat
status: active
date: 2026-04-21
origin: docs/brainstorms/2026-04-21-tts-abstraction-and-silero-qwen3-ab-requirements.md
---

# feat: TTS Provider Abstraction + Piper/Silero/Qwen3 A/B Experiment

## Overview

Introduce a TTS provider seam in `services/agent/` so `pipeline.py` selects between three engines (Piper, Silero v5, Qwen3-TTS) via config. Run a 3-way objective bench + conditional offline listening panel to decide whether one of the new engines replaces Piper as the Russian-language default, or Piper stays. Durable outputs: the seam, the adapters/sidecar, a reusable bench harness at `services/agent/scripts/tts_bench.py`, and a scorecard + decision under `docs/solutions/tts-selection/`.

## Problem Frame

Current agent TTS is Piper (`ru_RU-denis-medium`, CPU, wired at `services/agent/pipeline.py:117-121`). Team judgment is that the Russian voice sounds robotic, but no objective evidence exists. The brainstorm (see origin) narrowed candidates to three self-hosted Russian-capable, MIT/Apache-licensed, L4-fitting engines and committed to an abstraction-first approach so any future TTS swap becomes a config change. "Piper is fine, stay on it" is an explicit valid outcome.

## Requirements Trace

Full requirement list in origin doc. Grouped by this plan's implementation phases:

- **R21, R22, R23** — Phase 0 pre-commitment validation (premise blind-listen, Silero spike, Qwen3 wrapper smoke+license+vendor).
- **R1, R2, R3, R18** — provider selection seam + env-driven default, `PerSessionOverrides.voice` preservation.
- **R4, R5, R6, R7** — Silero v5 adapter + preload + torch cache.
- **R8, R8a, R8b, R9, R10** — Qwen3-TTS sidecar + wrapper pinning + dedicated HF cache + `OpenAITTSService` wiring.
- **R11, R11a, R11b, R12, R13, R14** — Russian eval corpus + 3-engine bench harness + UTMOS pinning + scorecard commit.
- **R15, R16, R17** — offline listening panel (conditional — only runs if pre-gate inconclusive).
- **R20** — keep the two non-winners for one release (7+3 days).

## Scope Boundaries

From origin Scope Boundaries, carried forward:

- No STT change (GigaAM-v3 stays).
- No LLM change.
- No multilingual TTS work (Russian-only; English voice deferred).
- No voice-cloning, no SSML/prosody steering, no cloud TTS baseline.
- **No in-SPA live-traffic A/B**, **no agent metrics export surface**, **no per-session TTS API field** — out of scope for this plan; see `### Deferred to Separate Tasks` for their re-consideration triggers.
- No full reusable LLM/STT/TTS benchmark rig (harness stays one-shot).
- No upstream PR for the Silero Pipecat adapter in this experiment (in-repo first; upstream is a follow-up).
- No tenant-facing changes (`tenants` table unchanged, admin API unchanged).

### Deferred to Separate Tasks

- **Silero TTS Pipecat adapter upstream PR** → follow-up task once the in-repo adapter is production-validated.
- **In-SPA live-traffic TTS A/B** → revisit once real playtester volume exists.
- **Agent Prometheus/metrics surface** → separate observability initiative; not required by offline-panel mode.
- **Cross-component benchmark rig** (LLM+STT+TTS reusable harness) → separate future commit.

## Context & Research

### Relevant Code and Patterns

- **Piper integration point:** `services/agent/pipeline.py:117-121` — `PiperTTSService(settings=PiperTTSService.Settings(voice=voice), download_dir=cfg.piper_voices_dir, use_cuda=False)`. Imported at line 31. Pipeline list at line 148-160.
- **AgentConfig:** frozen dataclass at `services/agent/pipeline.py:41-57` with `tts_voice`, `piper_voices_dir`. Add TTS engine field here.
- **Per-session override:** `services/agent/overrides.py:58-70` — `PerSessionOverrides.voice` exists as provider-specific string; no `tts_engine` field (do NOT add — per scope).
- **Model preload template:** `services/agent/models.py` (`load_gigaam`, `get_gigaam`, `_reset_models_for_tests`, module-global `_gigaam_model`). Startup orchestration at `services/agent/main.py:165-169` with hard-fail `sys.exit(3)` on load error.
- **Custom Pipecat TTSService template:** `services/agent/gigaam_stt.py` (STTService subclass, but the shape transfers directly). Key idioms: kwargs-only injection, `kwargs.setdefault(...)` for base defaults, `super().__init__(settings=STTSettings(...), **kwargs)` to silence `NOT_GIVEN`, eager model binding (not `_load()`), metrics context wrap, redacted ErrorFrame on exception.
- **Sidecar template:** `infra/docker-compose.yml:100-143` — `vllm` service. `x-gpu` YAML anchor (line 28-35), `profiles: [local-llm]` gating (line 107), loopback `127.0.0.1:port` bind (line 121), dedicated HF cache volume (line 114-116 explains UID-split rationale), `shm_size`, API-key-required via `:?` shell syntax.
- **Dockerfile.base:** `services/agent/Dockerfile.base` — multi-stage CUDA 13.2.1 + cuDNN + Python deps via `uv sync`. Runtime apt deps with one-line rationale each (line 71-75). Silero's torch should reuse `gigaam[torch]`'s pull — no double-pin.
- **Env var threading:** `services/agent/env.py`'s `require_env()` + `MissingEnvError`. `services/agent/main.py:142-157` reads all envs inside `try/except`, `sys.exit(2)` on miss. Envs → `_base_config` dict → `AgentConfig`.
- **Test pattern:** `services/agent/tests/test_gigaam_stt.py`. Stub heavy deps before imports (`sys.modules.setdefault("gigaam", _stub)` at module scope), sync tests + `asyncio.run()`, `_drain()` helper, `_build_service()` factory with `AsyncMock()` overrides for Pipecat framework hooks.
- **Existing pipecat extras pulled (verify in `services/agent/pyproject.toml`):** `pipecat-ai[livekit,piper,openai,silero]==0.0.108` — `[openai]` already present for `OpenAITTSService` (needed for Qwen3 wrapper), `[silero]` present for VAD only (TTS uses torch.hub directly).

### Institutional Learnings

- `docs/solutions/` is almost empty (one security doc, `xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md`). No prior TTS learnings. The frontmatter convention is the only precedent to mirror.
- Related: `docs/experiments/2026-04-19-stt-ab/phase-a-results.md` is referenced from `pyproject.toml:17` as a pointer to prior experimental artifacts. The TTS experiment follows the same pattern: scorecards + rating CSVs under `docs/solutions/tts-selection/`.

### External References

- Silero TTS v5 release (Oct 2025), MIT code + CC0/public-domain Russian weights: `https://github.com/snakers4/silero-models`. Python API: `torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v5_cis_base')` → object with `.apply_tts(text=..., sample_rate=48000)` returning a numpy waveform.
- Qwen3-TTS (Apache-2.0): `https://github.com/QwenLM/Qwen3-TTS`.
- Community OpenAI-compat wrapper: `https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi`. Must pin to specific commit SHA + vendor + license-check before R8 lands.
- UTMOS predictor (`sarulab-speech/UTMOS22-strong`): `https://huggingface.co/sarulab-speech/UTMOS22-strong`. Known to be EN/JA-trained; Russian scores are **directional-only**. Calibration against human labels is mandatory (see R13 + Open Questions).

## Key Technical Decisions

- **Factory function over Protocol/Registry class** — the simplest shape that satisfies R1 is a function `build_tts_service(engine: str, ...) -> TTSService` in `services/agent/tts_factory.py`. No abstract base class, no registry dict; just an if/elif dispatch that imports the relevant adapter lazily. Scope-guardian review flagged a generic TTSProvider Protocol as premature at MVP scale (see origin Outstanding Questions); this is the rationale for keeping it concrete.
- **Silero loads via `torch.hub`, cached to `/home/appuser/.cache/torch`** — no PyPI package exists for Silero v5 TTS. Mount a new `torch_cache_agent` named volume. Set `TORCH_HOME` env var in agent's compose service. Preload at startup, singleton in `models.py` alongside `gigaam`. (see origin: R4-R6)
- **Silero output resampled to 24 kHz via Pipecat's existing audio utilities** — Silero v5 `v5_cis_base` emits 24 kHz natively; LiveKit transport accepts 24 kHz output without resampling. Verify in R22 spike before committing. (see origin: R22)
- **Qwen3 via `OpenAITTSService` — no new agent-side adapter** — the community FastAPI wrapper exposes `/v1/audio/speech` (OpenAI contract), so Pipecat's upstream `OpenAITTSService` handles everything. Only new code on the agent side is the factory branch. (see origin: R9)
- **Qwen3 wrapper pinned + vendored + understood at `infra/qwen3-tts-wrapper/`** — never build from the upstream git URL directly. Dockerfile in `infra/qwen3-tts-wrapper/Dockerfile` clones at a specific commit SHA. Compose service builds from this local context. License recorded at `docs/solutions/tts-selection/wrapper-license.md`. **R23 adds an explicit "structural readthrough" step: before Unit 4 lands, the solo developer (with Claude Code) reads the entire wrapper source + deps and documents the request flow, dep pins, and CUDA/torch assumptions in `infra/qwen3-tts-wrapper/README.md` — so if upstream is abandoned post-flip, the wrapper is patchable in-house without cold-start. Pinning + vendoring alone does not mitigate upstream abandonment; the understanding step does.** (see origin: R8a)
- **`TTS_ENGINE` env var, no per-session override** — `TTS_ENGINE=piper|silero|qwen3` (default `piper`). Per-session routing via `POST /v1/sessions` body is out of scope (offline-panel mode doesn't need live-traffic routing). `PerSessionOverrides.voice` stays as an engine-specific voice-name string, interpreted by whichever engine `TTS_ENGINE` selects. (see origin: R18)
- **Harness is a plain `uv run` script, not a container** — `services/agent/scripts/tts_bench.py`. The agent image already has CUDA + torch + pipecat; the script uses the agent's venv. Containerize later only if flakiness demands it. R11a (VRAM-co-resident measurement) means the script runs from inside the agent container while the full stack is up (`docker exec`), not standalone on the host.
- **UTMOS22 pinned to `sarulab-speech/UTMOS22-strong` at a specific HF revision** — record the revision SHA in the harness. Calibration run (R13 Spearman ρ vs 5-10 human labels) happens inside Unit 5 (R13 lives under Unit 5's requirements) and records ρ in the bench scorecard.
- **Artifact naming** — three distinct markdown files to avoid "scorecard" ambiguity:
  - `docs/solutions/tts-selection/<YYYY-MM-DD>-bench-scorecard.md` — produced by Unit 5 (objective-bench output, UTMOS/TTFB/RTF/VRAM tables)
  - `docs/solutions/tts-selection/<YYYY-MM-DD>-decision.md` — produced by Unit 6 (cites bench-scorecard + applies success criteria + blind-sanity-check log)
  - `docs/solutions/tts-selection/panel-<YYYY-MM-DD>.md` — produced by Unit 7 (conditional offline panel)
- **Offline panel stores under `docs/solutions/tts-selection/`** — ratings as CSV under `panel/<YYYY-MM-DD>/`, analysis plan pre-registered at `tts-selection/analysis-plan.md`. Frontmatter (`problem_type: best_practice`).
- **"One release" retention = 10 calendar days total** — losing engines' code, weights, and compose entries stay in the tree for 7 working days post-flip; a deletion PR opens on day 7 and merges on day 10 after a 3-day QA pass. The two non-winning engines remain deployable via `TTS_ENGINE=<name>` the entire 10-day window.

## Open Questions

### Resolved During Planning

- **Qwen3 variant** — **0.6 B by default** (origin R8). 1.7 B evaluated only if 0.6 B underperforms AND R11a's co-resident VRAM measurement confirms it fits under production load.
- **Silero Python integration path** — `torch.hub.load('snakers4/silero-models', 'silero_tts', ...)`. No PyPI package. Separate `torch_cache_agent` volume; `TORCH_HOME` env.
- **Wrapper vendoring location** — `infra/qwen3-tts-wrapper/` (sibling to the existing `Caddyfile.prod` + compose configs in `infra/`).
- **Harness location** — `services/agent/scripts/tts_bench.py` (setting the precedent for a scripts directory).
- **Scorecard layout** — `docs/solutions/tts-selection/` with category-specific frontmatter (`problem_type: best_practice`, `category: tts-selection`, `module: services/agent/tts_factory`).
- **Current-Piper TTFB baseline measurement** — in-harness timing dump; not a rolling percentile (no metrics infra). Record median + p95 over the eval corpus.
- **`PerSessionOverrides.voice` semantics** — provider-specific string; each adapter interprets in its own namespace. Documented in the factory's docstring.
- **Factory abstraction shape** — concrete function, not Protocol/Registry. YAGNI; upgrade only if a fourth engine appears.

### Deferred to Implementation

- **Silero sample-rate behavior under barge-in** — R22 spike's primary unknown. If `v5_cis_base` emits at 24 kHz and file-at-a-time synth can be cancelled mid-buffer without leaving orphan audio in the LiveKit transport, scope is 1-2 days; if streaming or resampling is required, the Silero track expands. Spike output drives the decision.
- **Qwen3 wrapper license** — must be confirmed (MIT/Apache compatible) before Unit 4 merges. If incompatible, the Qwen3 track is dropped (R23) and the experiment reduces to Piper-vs-Silero.
- **Exact UTMOS torch version** — the UTMOS22 model may require a specific torch version that collides with what the agent image already has. Resolve during Unit 6 when the calibration run is attempted; may need a small `[tool.uv] override-dependencies` entry.
- **Phase 0 blind-listen cohort** — resolved: the team has 3 raters who participate in both R21 (Phase 0 premise listen) and Unit 7 (offline panel). Solo-eval not needed. Same 3 raters reused for the Unit 6 Decisive-winner sanity-check (2-of-3 agreement required to flip).
- **Exact `infra/qwen3-tts-wrapper/` Dockerfile shape** — depends on the wrapper's `requirements.txt` and whether it pulls torch itself or relies on a base image we supply. Determined during Unit 4.

## Output Structure

    services/agent/
      tts_factory.py                         # NEW — factory: build_tts_service(engine, ...) -> TTSService
      silero_tts.py                          # NEW — SileroTTSService Pipecat adapter
      models.py                              # MODIFIED — add load_silero / get_silero singletons
      pipeline.py                            # MODIFIED — replace PiperTTSService instantiation with factory call
      main.py                                # MODIFIED — read TTS_ENGINE + QWEN3_* envs, preload silero conditionally
      env.py                                 # MODIFIED — (no shape change) add TTS_ENGINE constant if helpful
      scripts/                               # NEW directory
        tts_bench.py                         # NEW — 3-engine objective harness (UTMOS, TTFB, RTF, VRAM)
        tts_panel.py                         # NEW — offline listening panel synth harness (conditional)
      tests/
        test_silero_tts.py                   # NEW — mirrors test_gigaam_stt.py structure
        test_tts_factory.py                  # NEW — factory resolution tests
        fixtures/tts_eval_corpus.csv         # NEW — 50-utterance Russian eval set with class labels

    infra/
      docker-compose.yml                     # MODIFIED — add qwen3-tts service + torch_cache_agent + hf_cache_qwen3 volumes
      .env.example                           # MODIFIED — add TTS_ENGINE, QWEN3_TTS_BASE_URL, QWEN3_TTS_API_KEY
      qwen3-tts-wrapper/                     # NEW — vendored community wrapper
        Dockerfile
        README.md                            # pinning rationale + upstream commit SHA
        LICENSE                              # copied from wrapper repo (verified compatible)

    docs/solutions/
      tts-selection/                         # NEW category directory
        analysis-plan.md                     # NEW — pre-registered decision rules (committed BEFORE panel runs)
        wrapper-license.md                   # NEW — Qwen3 wrapper license + pinning record
        2026-MM-DD-scorecard.md              # NEW — pre-gate scorecard (created by Unit 7)
        panel/2026-MM-DD/                    # NEW (conditional) — offline panel output if pre-gate is inconclusive
          blinded/<utterance-id>/{a,b,c}.wav
          _unblind.json                      # rater-inaccessible path
          ratings-<rater-id>.csv

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

**Factory seam (Unit 2):**

```text
# services/agent/tts_factory.py — sketch

def build_tts_service(engine: str, *, cfg: AgentConfig, voice: str) -> TTSService:
    match engine:
        case "piper":  return PiperTTSService(settings=..., download_dir=cfg.piper_voices_dir, use_cuda=False)
        case "silero": return SileroTTSService(silero_model=get_silero(), settings=SileroSettings(speaker=voice))
        case "qwen3":  return OpenAITTSService(base_url=cfg.qwen3_base_url, api_key=cfg.qwen3_api_key, model="qwen3-tts-0.6b", voice=voice)
        case _:        raise ValueError(f"unknown TTS_ENGINE: {engine}")
```

**Pipeline wiring (Unit 2, modified `pipeline.py`):**

```text
# before: tts = PiperTTSService(...)
# after:
tts = build_tts_service(cfg.tts_engine, cfg=cfg, voice=voice)
```

**Compose sidecar (Unit 4, added to `infra/docker-compose.yml`):**

```yaml
qwen3-tts:
  build:
    context: ./qwen3-tts-wrapper
  profiles: [tts-qwen3]
  <<: *gpu
  environment:
    - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    - HF_HOME=/root/.cache/huggingface
    - QWEN3_VARIANT=${QWEN3_VARIANT:-Qwen/Qwen3-TTS-0.6B}
  volumes:
    - hf_cache_qwen3:/root/.cache/huggingface
  ports:
    - "127.0.0.1:8002:8000"
  shm_size: "4gb"
  healthcheck:
    test: ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()\" || exit 1"]
    interval: 15s
    timeout: 5s
    start_period: 300s
```

**Success-criteria decision flow (Unit 7+):**

```text
Phase 0 (R21/R22/R23) -->
  premise-reject -> abort, no further work
  spike-blocker -> re-scope or drop that engine's track
  all-pass -> proceed to Units 2-6

Objective pre-gate (Unit 7) -->
  Decisive winner + blind sanity-check passes -> flip TTS_ENGINE, skip Unit 8
  Piper wins -> keep Piper, commit scorecard, close experiment
  Inconclusive -> Unit 8 (offline panel)
  All fail -> abort, commit scorecard, re-ideate

Offline panel (Unit 8) -->
  Significance + sanity-check -> flip TTS_ENGINE
  No significance -> keep Piper

Decision commit (Unit 9) -> scorecard doc + env flip (if any) + 7+3-day retention window
```

## Implementation Units

- [ ] **Unit 1: Phase 0 validation (premise listen + Silero spike + Qwen3 wrapper smoke)**

**Goal:** Three parallel timeboxed spikes that gate the entire experiment. Kill or re-scope the plan before committing to Units 2-9 if any fails.

**Requirements:** R21, R22, R23.

**Dependencies:** None (this unit gates all others).

**Files:**
- Create: `docs/solutions/tts-selection/phase-0-findings.md` (short memo: premise outcome, spike notes, wrapper smoke + license)
- Create: `infra/qwen3-tts-wrapper/` skeleton (clone wrapper at chosen commit SHA; `Dockerfile`, `README.md` documenting pin reason, `LICENSE` copied from upstream)

**Approach:**
- **R21 premise listen:** the 3-person team-rater cohort listens blind to 10 identical Russian utterances rendered by (a) current Piper output, (b) Silero v5, (c) Qwen3-TTS. **All three engines rendered through the same local synth path + identical utterances + same sample rate** (per review finding #4 — do not mix production-pipeline Piper capture with vendor cherry-picked demos; the confounded comparison invalidates the gate). Each rater independently ranks engines 1-3 per utterance. Aggregate. **Abort criterion: if fewer than 2 of 3 raters rank Piper last on a majority of utterances, treat the premise as unvalidated** and commit a "Piper acceptable for MVP" solution doc instead of proceeding. Ranking direction resolved: we are checking whether Piper is consistently the worst — `Piper-is-last-majority-of-utterances → proceed`.
- **R22 Silero spike:** 4-hour timebox. Clone `snakers4/silero-models`, run `torch.hub.load` to load `v5_cis_base`, inspect `.apply_tts()` output shape, sample rate, and timing. Confirm Pipecat's `TTSService` async-generator contract can be satisfied by wrapping a file-at-a-time synth (review `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/tts_service.py` for the required interface). Verify barge-in / `InterruptionFrame` handling: file-at-a-time means no mid-utterance interruption, so first-audio-after-interruption latency becomes synth-duration-dependent. **If the spike reveals a week-class issue (e.g., no way to yield partial frames, sample-rate incompatibility requires downstream refactor), re-scope — either accept the regression or drop the Silero track.**
- **R23 Qwen3 wrapper smoke + in-house understanding:** half-day timebox (up from 10-minute smoke). Three outputs:
  1. **Smoke** — pick a specific commit SHA of `dingausmwald/Qwen3-TTS-Openai-Fastapi`. Clone locally. Read its `LICENSE` (must be MIT/Apache-compatible — **blocking**). Confirm request/response shape matches Pipecat's `OpenAITTSService` expectation: `POST /v1/audio/speech` with `{model, input, voice}`, `response_format=pcm` supported, returns 24 kHz PCM stream. Run a single inference against `Qwen/Qwen3-TTS-0.6B` inside the agent container. **If wrapper is broken, unlicensed, or incompat, drop Qwen3 track — experiment proceeds Piper-vs-Silero.**
  2. **Variant-contract probes** — these are the items likeliest to quietly regress if the wrapper is ever abandoned: (a) what response sample-rate does the wrapper actually emit when the OpenAI contract says `pcm`? (b) does `voice=` pass through to Qwen3, or is it silently ignored? (c) what happens on unknown model name, unknown voice, oversized input? Document observed behavior, not vendor claims.
  3. **Structural readthrough** — the wrapper is small (expected ~100-300 lines of FastAPI). Solo developer + Claude Code reads through the full source + its `requirements.txt` + its Dockerfile, documents in `infra/qwen3-tts-wrapper/README.md`: the request flow (endpoint → Qwen3 call → response serialization), the dep pins, the CUDA/torch assumptions, and any non-obvious behaviors discovered in the probes. Goal is not to rewrite — it's to ensure that if upstream is later abandoned AND Qwen3 has won the experiment, the solo dev can patch the wrapper unassisted. Without this step, the vendored copy is a black box even though its source is in-tree.

**Execution note:** Each sub-spike is independent; run in parallel. Fail-fast — document the outcome in `phase-0-findings.md` and stop that track (not the whole plan) if the specific gate fails.

**Patterns to follow:**
- `docs/solutions/` frontmatter convention (see `docs/solutions/security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md`).
- Wrapper vendoring mirrors how `infra/Caddyfile.prod` lives alongside its runtime service, not pulled from external URL at build time.

**Test scenarios:**
- Test expectation: none — these are investigation spikes, not feature code. The deliverables are the memo + the vendored wrapper skeleton.

**Verification:**
- `docs/solutions/tts-selection/phase-0-findings.md` exists with explicit GO/NO-GO per engine track.
- If any track is NO-GO, downstream units are pruned accordingly in this plan.

---

- [ ] **Unit 2: TTS provider factory + pipeline wiring**

**Goal:** Introduce the `build_tts_service()` seam, thread `TTS_ENGINE` env through `main.py` → `AgentConfig` → `pipeline.py`, and replace the existing inline `PiperTTSService` instantiation with the factory call. No new engines yet — Piper-only dispatch works.

**Requirements:** R1, R2, R3, R18.

**Dependencies:** Unit 1 (premise listen must pass).

**Files:**
- Create: `services/agent/tts_factory.py`
- Modify: `services/agent/pipeline.py` (swap inline `PiperTTSService` at line 117-121 for `build_tts_service(...)`; add `tts_engine: str = "piper"` field to `AgentConfig`. **All new `AgentConfig` fields added by this plan default to `""` or `"piper"` to preserve dataclass ordering — Python's frozen dataclass requires default-less fields before defaulted ones, and the existing `tts_voice` field has no default.**)
- Modify: `services/agent/main.py` (read `TTS_ENGINE` env via `require_env("TTS_ENGINE", "piper")`, pass into `_base_config`)
- Modify: `infra/.env.example` (add `TTS_ENGINE=piper` with comment)
- Test: `services/agent/tests/test_tts_factory.py`

**Approach:**
- Factory signature: `build_tts_service(engine: str, *, cfg: AgentConfig, voice: str) -> TTSService`. Raise `ValueError` on unknown engine.
- Day-one the factory only knows `piper` — Silero and Qwen3 branches are added in Units 3 and 4 respectively. The factory file grows; `pipeline.py` stops growing.
- `AgentConfig.tts_engine: str = "piper"` default preserves behaviour if env is unset.
- Preserve the existing `voice = overrides.voice or cfg.tts_voice` line at `pipeline.py:73` — `voice` is passed through as-is; the factory interprets it per engine.

**Patterns to follow:**
- `AgentConfig` is `@dataclass(frozen=True)` (`pipeline.py:41`). Any new field must preserve immutability.
- Env threading pattern: `services/agent/env.py` + `services/agent/main.py:142-157`. New envs go inside the existing `try / except MissingEnvError` block; `sys.exit(2)` on miss (but `TTS_ENGINE` has a default so it won't miss).
- Factory is imported lazily in branches to keep `silero`/`qwen3` dependencies out of CPU-only test environments (match `services/agent/models.py`'s `# noqa: PLC0415` pattern).

**Test scenarios:**
- Happy path: `build_tts_service("piper", cfg=<test_cfg>, voice="ru_RU-denis-medium")` returns a `PiperTTSService` instance with the expected voice.
- Edge case: `build_tts_service("silero", ...)` / `build_tts_service("qwen3", ...)` — skip in this unit (deferred to Units 3/4 which import the relevant deps). Leave a `pytest.mark.skip(reason="added in Unit 3")` placeholder to make the coverage gap explicit.
- Error path: `build_tts_service("unknown", ...)` raises `ValueError` with the engine name in the message.
- Integration scenario: minimal pipeline build using the factory still produces a working `PipelineTask` with Piper as the TTS — the test can verify the `pipeline.py` changes didn't break the existing wire-up by re-using an existing integration test (if any) or by exercising `build_task(cfg=test_cfg)` and asserting the task builds without raising.

**Verification:**
- `TTS_ENGINE=piper` (or unset) produces identical behaviour to the current main branch — a live session with the default env runs Piper exactly as today.
- Factory raises `ValueError` on unknown engines at import time of the pipeline.

---

- [ ] **Unit 3: Silero v5 adapter + preload + torch cache volume**

**Goal:** Add `SileroTTSService`, the `load_silero()` / `get_silero()` singletons, agent-startup preload, the `torch_cache_agent` volume, and wire Silero into the factory.

**Requirements:** R4, R5, R6, R7.

**Dependencies:** Unit 1 (Silero spike), Unit 2 (factory).

**Files:**
- Create: `services/agent/silero_tts.py`
- Modify: `services/agent/models.py` (add `_silero_model`, `load_silero`, `get_silero`; extend `_reset_models_for_tests`)
- Modify: `services/agent/main.py` (preload `load_silero()` if `TTS_ENGINE=silero` — conditional preload keeps agent startup fast when Silero is not the active engine)
- Modify: `services/agent/tts_factory.py` (add `silero` branch)
- Modify: `services/agent/pyproject.toml` (no new top-level dep — Silero loads via `torch.hub`, which is already pulled by `gigaam[torch]`; add to runtime apt notes in Dockerfile.base if a missing system package surfaces during spike)
- Modify: `services/agent/Dockerfile.base` (set `ENV TORCH_HOME=/home/appuser/.cache/torch`; `mkdir -p /home/appuser/.cache/torch`; `chown appuser:appuser`)
- Modify: `infra/docker-compose.yml` (agent service: mount `torch_cache_agent:/home/appuser/.cache/torch`; add `torch_cache_agent:` to top-level `volumes:` block)
- Test: `services/agent/tests/test_silero_tts.py`

**Approach:**
- `SileroTTSService(TTSService)` subclass. Pipecat 0.0.108's `TTSService.run_tts` signature differs from `SegmentedSTTService.run_stt` — verify the exact shape in `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/tts_service.py` before implementing, and mirror the signature used by `OpenAITTSService.run_tts(self, text, context_id)` at `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/openai/tts.py`. Apply the GigaAMSTTService **idioms** (kwargs-only injection, eager model binding, `kwargs.setdefault("ttfs_p99_latency", 1.0)`, `super().__init__(settings=..., **kwargs)`) — but the frame contract is TTSService's (text in, audio-frame generator out), not STTService's.
- `run_tts(self, text, context_id)` wraps `await asyncio.to_thread(model.apply_tts, text=..., sample_rate=24000)` in `start_processing_metrics` / `stop_processing_metrics`, yields `TTSAudioRawFrame(audio=pcm, sample_rate=24000, num_channels=1)` with the `context_id` threaded through for the downstream LiveKit transport, catches exceptions → `ErrorFrame("TTS synth failed")` (redacted).
- `load_silero("v5_cis_base")` calls `torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v5_cis_base')`. Cached via `TORCH_HOME`.
- Conditional preload in `main.py`: only load Silero when `TTS_ENGINE=silero` — don't pay the load cost when Piper or Qwen3 is active.

**Execution note:** test-first against the Pipecat TTSService contract — write `test_silero_tts.py` mirroring `test_gigaam_stt.py` before the adapter body, with heavy `torch` / `silero` deps stubbed at module scope. This catches contract mismatches (sample rate, frame shape, interruption) before runtime.

**Patterns to follow:**
- `services/agent/gigaam_stt.py` for the TTSService subclass structure (adapted to the TTSService base).
- `services/agent/models.py` for the load/get/reset singleton trio.
- `services/agent/tests/test_gigaam_stt.py` for the test structure — autouse `_reset` fixture, `_drain()` helper, `_build_service()` factory with `AsyncMock` overrides of `start_processing_metrics` / `stop_processing_metrics`.

**Test scenarios:**
- Happy path: `SileroTTSService(silero_model=<mock>, settings=SileroSettings(speaker="v5_cis_base"))`, feed `"привет мир"`, assert a single `TTSAudioRawFrame` yielded with sample_rate=24000 and non-empty audio bytes.
- Edge case: empty text input → no audio frame yielded (or a zero-length frame — match the gigaam_stt.py precedent of early-return with no frame).
- Error path: `apply_tts` raises → `ErrorFrame("TTS synth failed")` yielded (redacted message asserted — exception detail must not leak to client, matching `gigaam_stt.py:171-177`).
- Error path: `get_silero()` called before `load_silero()` raises `RuntimeError("Silero model not loaded — call load_silero() first")`.
- Edge case: non-Russian language requested → the adapter rejects with a clear error (Russian-only per R4) or logs a warning and falls back to ru.
- Integration scenario: factory branch — `build_tts_service("silero", cfg=<test_cfg>, voice="v5_cis_base")` returns a `SileroTTSService` with the injected model singleton (mocked).

**Verification:**
- `TTS_ENGINE=silero` deploy boots the agent (preload succeeds, model cached in `torch_cache_agent` volume), `POST /v1/sessions` creates a live session, TTS output audible + correctly sampled at 24 kHz, no Pipecat-framework warnings (STTSettings/NOT_GIVEN equivalent) on pipeline build.
- `docker compose down` and back `up` — second boot hits the cached weights, load time noticeably shorter.

---

- [ ] **Unit 4: Qwen3-TTS sidecar (vendored wrapper + compose + env wiring)**

**Goal:** Ship the Qwen3 track as a compose sidecar. Vendor the wrapper, build an image from the local context, add dedicated HF cache volume, wire env vars through the agent, add the `qwen3` branch to the factory. No new agent-side Pipecat adapter — `OpenAITTSService` handles the protocol.

**Requirements:** R8, R8a, R8b, R9, R10.

**Dependencies:** Unit 1 (Qwen3 wrapper smoke + license check), Unit 2 (factory).

**Files:**
- Create/expand: `infra/qwen3-tts-wrapper/Dockerfile` (skeleton from Unit 1 → functional)
- Create: `infra/qwen3-tts-wrapper/README.md` (pinning rationale, upstream commit SHA, how to bump)
- Modify: `infra/docker-compose.yml` (add `qwen3-tts` service under `tts-qwen3` profile, add `hf_cache_qwen3:` volume)
- Modify: `infra/.env.example` (add `QWEN3_TTS_BASE_URL`, `QWEN3_TTS_API_KEY`, `QWEN3_VARIANT=Qwen/Qwen3-TTS-0.6B`)
- Modify: `services/agent/pipeline.py` (extend `AgentConfig` with `qwen3_base_url: str = ""`, `qwen3_api_key: str = ""` — defaulted to preserve frozen dataclass field ordering; factory branch validates they're non-empty at dispatch time when `TTS_ENGINE=qwen3`)
- Modify: `services/agent/main.py` (read `QWEN3_TTS_BASE_URL`, `QWEN3_TTS_API_KEY` envs via `require_env`; pass through `_base_config`)
- Modify: `services/agent/tts_factory.py` (add `qwen3` branch — returns `OpenAITTSService(base_url=cfg.qwen3_base_url, api_key=cfg.qwen3_api_key, model="qwen3-tts-0.6b", voice=voice)`)
- Create: `docs/solutions/tts-selection/wrapper-license.md` (license record — authored in Unit 1, finalized here)
- Test: `services/agent/tests/test_tts_factory.py` — un-skip the Qwen3 branch test

**Approach:**
- Sidecar uses the same profile-gated + loopback-bind + dedicated-cache pattern as `vllm`. `<<: *gpu` anchor applied. `shm_size: "4gb"` is sufficient for a 0.6B model (vs vLLM's 16 GB).
- `QWEN3_VARIANT` env at the sidecar level lets us flip between 0.6 B and 1.7 B without changing agent code (honors R10). Default is 0.6 B.
- Agent talks to `http://qwen3-tts:8000/v1/audio/speech` over the internal docker network. The compose host port `127.0.0.1:8002:8000` exists only for debugging/curl-probing from the host.
- `QWEN3_TTS_API_KEY` is set to a generated secret (compose `${QWEN3_TTS_API_KEY:?...}`) — agent sends it; wrapper validates. Prevents accidentally-exposed wrapper from being callable.
- Healthcheck copied from the `vllm` pattern (`urllib.request.urlopen('http://localhost:8000/health')` + `start_period: 300s`).

**Patterns to follow:**
- `infra/docker-compose.yml:100-143` (vllm service) for every structural field.
- `infra/docker-compose.yml:114-116` comment justifies the per-service HF cache split — **this is load-bearing for Qwen3-TTS too. Do not share `hf_cache_agent`.**

**Test scenarios:**
- Happy path: `build_tts_service("qwen3", cfg=<test_cfg with qwen3_*>, voice="alloy")` returns an `OpenAITTSService` instance with the expected `base_url` and `api_key`.
- Edge case: missing `QWEN3_TTS_BASE_URL` env at startup when `TTS_ENGINE=qwen3` → `sys.exit(2)` via `MissingEnvError` (do NOT exit when `TTS_ENGINE=piper` — the envs are optional for non-qwen3 runs).
- Integration scenario (manual, not automated): `docker compose --profile tts-qwen3 up`, `curl http://localhost:8002/health` returns 200 after start_period, `docker compose logs qwen3-tts` shows model loaded.
- Integration scenario (manual): with `TTS_ENGINE=qwen3`, a real session produces audible Russian TTS via the sidecar path.

**Verification:**
- `docker compose --profile tts-qwen3 up` brings the sidecar up healthy; agent with `TTS_ENGINE=qwen3` dispatches to the sidecar and receives streaming audio.
- `docker compose down && up` without `--profile tts-qwen3` skips the sidecar entirely (agent with `TTS_ENGINE=piper` unaffected).
- Wrapper license recorded at `docs/solutions/tts-selection/wrapper-license.md` with the pinned commit SHA.

---

- [ ] **Unit 5: Russian eval corpus + objective bench harness**

**Goal:** Commit the 50-utterance Russian corpus with class labels, build `services/agent/scripts/tts_bench.py` that runs all three engines through the factory, measures UTMOS + TTFB + RTF + VRAM-co-resident, emits a scorecard. Calibrate UTMOS against 5-10 human labels.

**Requirements:** R11, R11a, R11b, R12, R13, R14.

**Dependencies:** Units 2, 3, 4 (factory + both new engines wired).

**Files:**
- Create: `services/agent/tests/fixtures/tts_eval_corpus.csv` (50 utterances, columns: `utterance_id`, `text_ru`, `class` ∈ {short, medium, long, domain})
- Create: `services/agent/scripts/tts_bench.py`
- Create: `docs/solutions/tts-selection/utmos-calibration.md` (Spearman ρ record, pinned UTMOS revision SHA)
- Modify: `services/agent/pyproject.toml` (if UTMOS requires a specific torch pin; likely needs a `[tool.uv] override-dependencies` entry)

**Approach:**
- Corpus is hand-authored: ~12 short (≤5 words), ~15 medium (~15 words), ~15 long (≥40 words), ~8 domain (game vocabulary, character/NPC names, Russian-specific phonetic challenges). Committed as plain CSV — text-only, no audio.
- Harness flow per engine:
  1. Resolve the engine via the factory (reusing the real production seam).
  2. For each utterance: synth, capture PCM buffer + wall-clock time-to-first-audio + wall-clock total-synth-time, compute RTF = synth_time / audio_duration.
  3. Score each audio buffer with UTMOS (pinned model rev).
  4. Sample VRAM via `nvidia-smi --query-gpu=memory.used` or `torch.cuda.memory_allocated()` just before/after synth — **R11a: must run under `docker exec` into the running agent container with `--profile local-llm --profile tts-qwen3` active so VRAM reading reflects production co-residency**.
  5. Emit scorecard markdown with per-engine + per-class tables.
- R11b (current-Piper TTFB baseline) drops out of the same harness run: Piper is one of the engines, so its TTFB numbers appear in the scorecard alongside the others.
- R13 calibration: pick 5-10 utterances from the corpus, synthesize on each engine, 1 human rater assigns MOS 1-5 per clip blindly. Compute Spearman ρ between UTMOS scores and human scores. If ρ < 0.6, flag in the scorecard that UTMOS should be treated directionally only (reinforces the origin Dependencies assumption).

**Execution note:** Harness is standalone; no pytest integration. Run via `docker exec agent uv run python services/agent/scripts/tts_bench.py --engines piper,silero,qwen3 --corpus services/agent/tests/fixtures/tts_eval_corpus.csv --out docs/solutions/tts-selection/$(date +%Y-%m-%d)-scorecard.md`.

**Patterns to follow:**
- Scorecard frontmatter mirrors `docs/solutions/security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md` shape (category, module, problem_type=best_practice, tags, etc.).
- UTMOS pin: record HF revision SHA, not just the model name. Matches the `gigaam[torch] @ git+...@<sha>` convention in `pyproject.toml`.

**Test scenarios:**
- Happy path: harness script runs end-to-end against a short test corpus (3 utterances, 1 engine), emits a valid markdown scorecard.
- Edge case: engine raises during synth → harness catches, marks that (engine, utterance) as `error` in the scorecard, continues with remaining cells.
- Edge case: UTMOS model fails to load → harness fails early with a clear error (before any synth), so we don't waste time generating audio we can't score.
- Integration scenario: full run with all 3 engines against the 50-utterance corpus produces per-class tables and a VRAM-co-resident reading ≠ 0 (confirms R11a contract).
- Test expectation for the harness itself: unit tests for the corpus loader + scorecard formatter live in `test_tts_bench.py` (not enumerated here — small helpers, add as convenient).

**Verification:**
- Running the harness produces `docs/solutions/tts-selection/<date>-scorecard.md` with populated tables for all 3 engines, per-class UTMOS breakdowns, TTFB median + p95, RTF, VRAM-co-resident.
- Spearman ρ recorded in `utmos-calibration.md`.

---

- [ ] **Unit 6: Pre-gate decision — interpret scorecard, apply criteria**

**Goal:** Apply the following Success Criteria to the Unit 5 bench-scorecard. Produce a decision artifact: Decisive winner + sanity-check, Piper wins, Inconclusive → trigger Unit 7, or All fail → abort.

**Inline Success Criteria (from origin doc — reproduced here so Unit 6 is self-contained):**
- **Decisive winner** — one engine's mean UTMOS is ≥ 0.3 higher than **both** other engines AND its TTFB p95 is ≤ 500 ms on L4 AND no utterance class drops below 2.5 UTMOS AND its end-to-end first-audio p95 does not regress vs measured-Piper baseline (from R11b) by more than 100 ms. Final gate: **mandatory 2-rater blind 10-sample sanity-check** before flipping `TTS_ENGINE`.
- **Piper wins** — Piper's mean UTMOS is within 0.2 of the best other engine AND Piper's subjective sanity-check (same 2-rater gate) is acceptable. Keep Piper, close the experiment.
- **Inconclusive** — no single engine is ≥ 0.3 above both others, OR two engines tie within 0.2 above Piper. Proceed to Unit 7 (offline panel).
- **All fail** — every engine's UTMOS falls below Piper on ≥ 2 utterance classes. Abort, commit findings, re-ideate.

**Requirements:** Origin Success Criteria (Objective pre-gate).

**Dependencies:** Unit 5.

**Files:**
- Modify: `docs/solutions/tts-selection/<date>-scorecard.md` (append decision section + outcome)
- Create (if Decisive winner): `docs/solutions/tts-selection/decision.md` pointing to chosen engine + flip date
- Create (if Decisive winner): blind-sanity-check log — `docs/solutions/tts-selection/blind-sanity-check.md` with rater names + decision

**Approach:**
- Read the scorecard. Apply the 4 Success Criteria branches from the origin doc (Decisive winner / Piper wins / Inconclusive / All fail).
- On Decisive winner: run the mandatory ≥ 2-rater blind 10-sample sanity-check. 2 team members each listen to 10 random samples from the corpus synthesized by the winner; confirm naturalness matches the UTMOS ranking. Log the raters, the samples, and the verdict at `blind-sanity-check.md`. If any rater rejects, demote to Inconclusive and proceed to Unit 7.
- On Piper wins: commit the scorecard outcome, close the experiment. Units 7-9 skipped.
- On Inconclusive: trigger Unit 7.
- On All fail: commit the outcome, abort, re-ideate separately.

**Execution note:** This is a human-in-the-loop unit. No code. The deliverable is the decision artifact + any gate artifacts.

**Patterns to follow:**
- Scorecard append convention: add a `## Decision` section with explicit reference to the origin Success Criteria branch.

**Test scenarios:**
- Test expectation: none — this is an interpretation + decision step, not code.

**Verification:**
- A `## Decision` section exists in the scorecard with one of the four outcomes explicitly cited. If Decisive winner, the blind-sanity-check log exists and is referenced.

---

- [ ] **Unit 7: Offline listening panel (conditional — runs only if pre-gate is Inconclusive)**

**Goal:** Build the offline panel harness, synthesize a blinded corpus, collect independent rater CSVs, aggregate with Wilcoxon signed-rank, produce a panel scorecard. Required pre-registered analysis plan must exist **before** any clip is synthesized.

**Requirements:** R15, R16, R17.

**Dependencies:** Unit 6 returning "Inconclusive".

**Files:**
- Create (before synth): `docs/solutions/tts-selection/analysis-plan.md` (decision rule, exclusion criteria, tie-breakers, rater list)
- Create: `services/agent/scripts/tts_panel.py` (synth harness)
- Create: `docs/solutions/tts-selection/panel/<YYYY-MM-DD>/blinded/<utterance-id>/{a,b,c}.wav`
- Create: `docs/solutions/tts-selection/panel/<YYYY-MM-DD>/_unblind.json` (rater-inaccessible)
- Create: `docs/solutions/tts-selection/panel/<YYYY-MM-DD>/ratings-<rater-id>.csv` (one per rater)
- Create: `docs/solutions/tts-selection/panel-<YYYY-MM-DD>.md` (aggregated scorecard)

**Approach:**
- Pre-register the analysis plan **first**. At minimum: decision threshold (e.g., Wilcoxon p < 0.05 on engine-pair using the **per-utterance mean across 3 raters** as the paired-observation unit — effective n = 30 utterance-pairs, which clears the Wilcoxon minimum-p floor), exclusion rule (ratings from clips marked as technical failure are dropped), tie-breaker (if both engine-pair tests fail significance, Piper wins by default), rater cohort (3-person team documented in analysis-plan.md).
- Harness picks ≥ 30 utterances from the corpus, synthesizes on each surviving candidate, writes blinded on-disk layout with random labels per utterance (different random mapping per utterance to avoid a consistent `a=piper` leak).
- `_unblind.json` stored outside the rater-facing directory (e.g., at the panel date root, not under `blinded/`).
- Raters score independently — no Slack discussion, no shared docs. Commit `ratings-<rater-id>.csv` with columns (`utterance_id`, `blinded_label`, `score_1_5`, `failure_note`).
- Aggregation: per-engine median + IQR + Wilcoxon signed-rank for each engine-pair. Results committed as `panel-<date>.md` referencing the analysis plan.

**Execution note:** Pre-registration discipline is the whole point of this unit. If the analysis plan is modified after synth but before aggregation, that's a protocol violation — re-run with the new plan and a fresh corpus sample.

**Patterns to follow:**
- Frontmatter per `docs/solutions/` convention.
- Offline-synth flow mirrors `tts_bench.py` but writes WAVs instead of scoring in-process.

**Test scenarios:**
- Happy path: panel harness runs against a mini-corpus (3 utterances × 2 engines), produces blinded directory + unblind JSON, with each label actually corresponding to its intended engine.
- Edge case: raters disagree wildly → Wilcoxon returns p ≫ 0.05 on all pairs → pre-registered tie-breaker fires (Piper wins).
- Integration scenario: full panel run + 3 rater CSVs + aggregation → panel scorecard committed with a conclusion.
- Test expectation for the unit's code: harness blinding correctness (no `a=` constant across utterances), tested with a seeded-random run.

**Verification:**
- `analysis-plan.md` timestamp precedes the first synth timestamp in git log.
- `panel-<date>.md` cites a concrete decision under the pre-registered rules.
- If the panel names a non-Piper winner, a blind-sanity-check (same rules as Unit 6) runs before any env flip.

---

- [ ] **Unit 8: Flip `TTS_ENGINE` default + 7+3-day retention window**

**Goal:** Apply the decision. If a non-Piper engine won, flip the `TTS_ENGINE` default in `infra/.env.example` and in prod deployments. Open the retention-window calendar marker so the losing engines' code isn't deleted for 7+3 days.

**Requirements:** R20.

**Dependencies:** Unit 6 (decisive winner) OR Unit 7 (panel winner) — or both skipped if Piper wins.

**Files:**
- Modify: `infra/.env.example` (flip `TTS_ENGINE` default if non-Piper won)
- Modify: `infra/terraform-gcp/` — specifically the secret that feeds `infra/.env` on instance boot. The env-var chain per CLAUDE.md is: Terraform → SSM Parameter Store → instance IAM role → `user_data.sh` fetches at boot → writes `infra/.env`. Update the `TTS_ENGINE` entry in whichever Terraform resource holds the SSM-parameter set (confirm exact file during Unit 8; `infra/terraform-gcp/main.tf` is the likely entry). Instance restart picks up the new value on next boot.
- Create: `docs/solutions/tts-selection/final-decision.md` (final summary, winner, retention-deletion-date = flip-date + 10)
- Create: tracking issue / calendar reminder for the deletion PR

**Approach:**
- If Piper won: no code flip. Commit final-decision.md noting Piper stays, close the experiment, proceed to delete Silero + Qwen3 branches + sidecar + vendored wrapper + cache volumes on day 10 (tracked separately).
- If a new engine won: update `infra/.env.example` default, bump prod secret, redeploy agent. Schedule deletion PR for day 10.
- Retention window: the losing engines' code stays in the tree for 7 calendar days after the flip; the solo developer opens the deletion PR on day 7 and merges it on day 10 after a 3-day QA pass. Solo-dev ops — no formal review SLA; the developer self-reviews via Claude Code before merge.
- The deletion PR is out of scope for this plan — it's a separate commit after the retention window closes. Self-assigned calendar reminder on day 7.

**Patterns to follow:**
- Env flip mirrors how `TTS_VOICE=ru_RU-denis-medium` is set in `infra/.env.example`.
- Secret-flip via `infra/terraform-gcp/` follows the prod-secret-bump pattern in `docs/brainstorms/2026-04-19-voice-stack-ab-experiment-requirements.md` (though the original STT env plumbing was cleaned up — treat this as the first new env-flip since cleanup).

**Test scenarios:**
- Test expectation: none — this is a config flip + documentation commit.

**Verification:**
- `TTS_ENGINE=<winner>` active in prod; a live session produces the winner's audio output; Unit 1's Piper baseline preserved in git history for rollback.
- Final decision doc exists; retention deletion date is concrete (YYYY-MM-DD) and tracked.

## System-Wide Impact

- **Interaction graph:** TTS is one node in the Pipecat pipeline (`pipeline.py:148-160`). The seam introduced in Unit 2 is invisible to every other node — `transport.input`, VAD, STT, LLM, aggregators are unaffected. Barge-in (recently fixed via `VADUserTurnStartStrategy`) interacts with TTS only through `InterruptionFrame` propagation; the new engines must respect the existing interruption contract.
- **Error propagation:** TTS errors yield `ErrorFrame` downstream (redacted message per the existing convention in `gigaam_stt.py:171-177`). Upstream: a Qwen3 sidecar failure surfaces as a Pipecat framework-level exception on the `OpenAITTSService` connection; the pipeline retries per its existing behaviour, no new retry logic added.
- **State lifecycle risks:** Silero's singleton cache in `models.py` persists across sessions (same as GigaAM today). `_reset_models_for_tests()` must clear it alongside gigaam. Qwen3 has no agent-side state — all state is sidecar-local.
- **API surface parity:** No external API change. Internal env contract expands (`TTS_ENGINE`, `QWEN3_TTS_BASE_URL`, `QWEN3_TTS_API_KEY`) — all with sane defaults so `TTS_ENGINE=piper` deployments behave identically to today.
- **Integration coverage:** The factory's branch-per-engine means a real live-session test per engine is the only way to confirm end-to-end behavior. Unit tests can mock the factory; they cannot prove the LiveKit transport accepts each engine's sample rate without a pipeline build + dispatch.
- **Unchanged invariants:**
  - `PerSessionOverrides.voice` contract is preserved (no new per-session TTS API field — out of scope).
  - `tenants` table, admin API, webhook surface unchanged.
  - VAD + STT + LLM + barge-in behavior identical to today.
  - `hf_cache_agent` volume semantics preserved; new volumes (`torch_cache_agent`, `hf_cache_qwen3`) are additive.
  - Current Piper behavior survives unchanged under `TTS_ENGINE=piper` (the pre-Unit-2 default).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| R22 Silero spike reveals Pipecat `TTSService` contract mismatch (sample rate, streaming, barge-in) | Unit 1 is explicitly a 4-hour timeboxed spike with go/no-go. Re-scope or drop the track if discovered. |
| R23 Qwen3 community wrapper is abandoned or licence-incompatible (Unit 1) | Pin commit SHA, vendor into repo, check license — drop the track if wrapper fails at R23. |
| R23 Qwen3 wrapper is abandoned *after* flip (post-experiment, Qwen3 won) | R23 "structural readthrough" produces `infra/qwen3-tts-wrapper/README.md` documenting the full request flow, dep pins, CUDA assumptions — so the solo dev can patch it unassisted if upstream rots 60+ days out. |
| Qwen3-1.7B + vLLM-7B co-residency OOM | R11a's harness measures VRAM under production load; 0.6B is the committed default, 1.7B only enabled if measurement confirms headroom. |
| `torch.hub.load` requires network at first boot (Silero) | `torch_cache_agent` named volume caches weights after first load; docker-compose `start_period: 300s` on healthcheck tolerates first-run download. |
| UTMOS Russian validity is low (Spearman ρ < 0.6) | R13 calibration step records ρ; scorecard flags low-ρ runs as directional-only; subjective panel becomes final arbiter. |
| Offline panel raters disagree or refuse to participate | Pre-registered analysis plan includes tie-breaker (Piper wins on insufficient signal); experiment aborts gracefully instead of forcing a conclusion. |
| Losing engine code lingers post-retention window, accumulating tech debt | Retention deletion PR tracked as a calendar event on day 7; plan acknowledges Unit 8 schedules the cleanup PR but the cleanup itself is out of scope. |
| Blind-label mapping leaks (rater discovers `a=silero` across utterances) | Unit 7 uses per-utterance random mapping (not a per-session constant); `_unblind.json` stored outside rater-facing path. |
| Multiple parallel pre-gate runs cause VRAM contention | Harness documents `--profile local-llm --profile tts-qwen3` as the expected environment; serial execution per the current single-L4 constraint. |

## Documentation / Operational Notes

- `CLAUDE.md` updates: add a new section or update the existing TTS bullet documenting the factory seam + env selector. Keep it brief (3-5 lines).
- `infra/.env.example` grows by ~3 env vars (`TTS_ENGINE`, `QWEN3_TTS_BASE_URL`, `QWEN3_TTS_API_KEY`) — each with a comment explaining the value domain.
- `docs/solutions/tts-selection/` becomes the permanent home for TTS-related institutional knowledge. Final-decision doc links back to this plan.
- Agent image rebuild required when Dockerfile.base changes (Unit 3's `TORCH_HOME` + cache-dir mkdir). This is a heavy rebuild — schedule for a low-traffic window.
- Deployment: new compose volumes (`torch_cache_agent`, `hf_cache_qwen3`) auto-create on first `docker compose up`. No terraform changes needed.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-21-tts-abstraction-and-silero-qwen3-ab-requirements.md](../brainstorms/2026-04-21-tts-abstraction-and-silero-qwen3-ab-requirements.md)
- Related code:
  - `services/agent/pipeline.py:117-121` (Piper integration point)
  - `services/agent/models.py` (load/get singleton pattern)
  - `services/agent/gigaam_stt.py` (custom Pipecat service subclass pattern)
  - `services/agent/tests/test_gigaam_stt.py` (test pattern to mirror)
  - `infra/docker-compose.yml:100-143` (sidecar template: vllm service)
  - `infra/docker-compose.yml:114-116` (HF cache per-service split rationale)
  - `services/agent/Dockerfile.base` (multi-stage CUDA + Python deps build)
- Related prior art:
  - `docs/brainstorms/2026-04-19-voice-stack-ab-experiment-requirements.md` (STT A/B pattern reused for the 3-way comparison)
  - `docs/experiments/2026-04-19-stt-ab/phase-a-results.md` (referenced in `pyproject.toml:17` — precedent for experiment output placement)
  - `docs/solutions/security-issues/xff-spoof-and-shared-cache-eviction-in-ip-rate-limiter-2026-04-20.md` (frontmatter template for solution docs)
- External docs:
  - Silero TTS v5 — https://github.com/snakers4/silero-models
  - Qwen3-TTS — https://github.com/QwenLM/Qwen3-TTS
  - Community OpenAI-compat wrapper — https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi (pin by commit SHA in Unit 1)
  - UTMOS22 predictor — https://huggingface.co/sarulab-speech/UTMOS22-strong (pin by revision in Unit 5)
  - Pipecat `TTSService` — `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/tts_service.py` (reference for the async-generator contract)

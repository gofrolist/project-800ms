---
date: 2026-04-23
topic: fastcosyvoice-fifth-tts-engine
---

# FastCosyVoice as a Fifth TTS Engine

## Problem Frame

The agent already wires four pluggable TTS engines through `services/agent/tts_factory.py` — Piper (CPU default), Silero v5 (GPU, Russian, non-streaming), Qwen3-TTS (OpenAI-compat sidecar, currently **non-streaming on L4** per `services/agent/qwen3_tts.py:115-131` — buffer underruns at RTF ~3x), XTTS v2 (in-process, 17 languages, CPML non-commercial). None of them combine: (a) streaming audio output that actually survives L4 RTF budget, (b) Apache-2.0-licensed weights cleared for commercial use, (c) <800 ms first-audio latency on a single L4 GPU, (d) native voice control via reference audio — though the cross-lingual path for (d) has known quality-artifact risks on CV3 (Chinese prosody leaking into RU output).

[Brakanier/FastCosyVoice](https://github.com/Brakanier/FastCosyVoice) is a ground-up re-implementation targeting `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` with TensorRT optimization, Russian stress handling via `silero-stress`, Cyrillic normalization fixes, a 30 s reference-audio extension, and a `uv`-native install. The author reports **0.797 s TTFB on RTX 3090** with TRT-Flow + `torch.compile`. L4 has ~70 % higher FP16 Tensor throughput than 3090 but only ~32 % the memory bandwidth (300 GB/s vs 936 GB/s); LLM prefill is bandwidth-bound, DiT flow is compute-bound — net direction on L4 is **unknown and must be measured** before scope is committed.

**Why now.** Neither an in-tree latency regression nor a user complaint drives this: Piper and Silero already ship RU voice. This work is a capability-completion bet — making a single Apache-2.0 engine with zero-shot voice control and streaming available, so a future voice-identity / tenant-branded-voice product move isn't blocked on XTTS's CPML license or Piper's phoneme-quality ceiling. Framing as a **voice-assistant feature** (not a TTS-platform expansion) — engine count is incidental; the acceptance bar is user outcome on a single voice.

Adding it as a **fifth engine alongside the existing four** preserves the operator workflow — env flip `TTS_ENGINE=fastcosy` or per-session `overrides.tts_engine`. The brainstorm is inherently architectural: the engine cannot be installed in-process (it pins `requires-python = ">=3.10,<3.12"`, the agent is 3.12) and pins CUDA 12.8 wheels while the agent runs CUDA 13.2 — sidecar shape is forced, not chosen. If the measurement/blind-listen gates below do not pass, v1 is deferred and `fastcosy` is not shipped as a selectable engine; existing four stay untouched.

## Requirements

**Surface and integration shape**

- **R1.** FastCosyVoice is exposed as a new `fastcosy` engine value. This requires **four synchronized declaration sites** (enforced by `services/agent/tests/test_engine_parity.py`): (a) `services/agent/tts_factory.py::build_tts_service` dispatch branch, (b) `services/agent/pipeline.py::_VALID_TTS_ENGINES` frozenset, (c) `services/agent/overrides.py::_VALID_TTS_ENGINES` frozenset, (d) `apps/api/schemas.py::TtsEngine` Literal. Selection happens via the existing `TTS_ENGINE` env var or per-session `overrides.tts_engine` payload.
- **R2.** The engine runs as a **vendored sidecar** under `infra/fastcosy-tts-wrapper/`, following the provenance, patch-in-place, and selective-copy conventions already documented in `infra/qwen3-tts-wrapper/README.md`. The agent process never imports FastCosyVoice directly. A `infra/fastcosy-tts-wrapper/PROVENANCE.md` (or README.md section) records: upstream URL, vendored commit SHA, fetch date, and a SHA256 of the vendored tarball, so provenance is detectable in-tree and upstream drift is caught on re-sync. Any re-sync is a separate PR that updates these fields.
- **R3.** The sidecar exposes an **OpenAI-compatible HTTP API** (`POST /v1/audio/speech`), with 24 kHz mono int16 PCM output (matching the repo-wide standard; any native-rate resample happens inside the sidecar). The agent-side adapter subclasses Pipecat's upstream `OpenAITTSService` following the `services/agent/qwen3_tts.py` pattern — a ~85-line `run_tts` override (not a "thin error-redaction subclass") that catches `httpx` network exceptions and redacts to `ErrorFrame`. The sidecar's error response format must match Qwen3's: 4xx/5xx with a JSON body the agent's redaction logic can parse. **Streaming (`stream=True` via `extra_body`) is gated on the L4 RTF measurement in Resolve-Before-Planning Q1** — if measured RTF ≥ 1.0x with no pre-roll budget, the adapter ships in non-streaming mode (full-utterance buffer, matching `qwen3_tts.py`'s current production behavior) and the Latency Success Criteria is re-derived accordingly.

**Voice model and content**

- **R5.** Voice identity is delivered via the repo's existing `clone:<profile>` voice-string convention. The sidecar resolves the profile name against the existing shared voice library at `/opt/voice_library/profiles/<profile>/` — matching the layout already documented in `voice_library/README.md` and consumed by `services/agent/xtts_tts.py::_resolve_voice_profile` and the Qwen3 sidecar. Each profile directory contains `meta.json` (with `ref_audio_filename`, `ref_text`, `language`, and an optional `instruction` field for FastCosyVoice's cross-lingual prompt prefix) and the reference audio file named by `ref_audio_filename`. Reference transcript lives inside `meta.json` (`ref_text`) — not as a separate `ref.txt` sibling — so the layout stays engine-agnostic.
- **R6.** v1 ships **one default profile** (`fastcosy_default`) using a Chinese reference clip + transcript — the cross-lingual workaround per upstream maintainer guidance on [`FunAudioLLM/CosyVoice#1790`](https://github.com/FunAudioLLM/CosyVoice/issues/1790). **Audition is bounded**: at most **N=3 candidate clips** (starting with FastCosyVoice's `asset/zero_shot_prompt.wav` and `refs/019.wav` from the author's demo) are evaluated against R21 before selection. If none of the N clips passes the quality gate, **v1 is deferred and `fastcosy` does not ship as a selectable engine** — no unbounded audition loop.
- **R7.** Russian text is normalized before synthesis via FastCosyVoice's built-in `auto_stress` flag (silero-stress integration). **Default is enabled**, sidecar-side env toggle (`FASTCOSY_AUTO_STRESS=false`) disables it. Manual `+`-before-vowel stress syntax is passed through untouched. **Collision risk with LLM output** (mixed EN/RU code-switching, already-stressed text, numerical expressions) is not yet verified — see Resolve-Before-Planning Q3. If the corpus probe shows `auto_stress` crashes or misbehaves on representative LLM output, the default flips to `false`.

**Sidecar-internal configuration** (not agent-facing)

- **R8.** `FASTCOSY_LLM_WEIGHTS` toggles between `llm.pt` (baseline) and `llm.rl.pt` (RL-post-trained, **3.79 % RU WER vs 6.77 %** per CV3 paper Table 5). Default `llm.rl.pt`.
- **R9.** `FASTCOSY_MODE` accepts `basic` | `trt-flow` (default, v1 ships this) | `trt-llm` (reserved, errors at startup). Mode B (`trt-flow`) is fp16 + TensorRT Flow (DiT) + `torch.compile` for the LLM. `basic` emits a startup WARNING (`FastCosyVoice basic mode — not production-safe`) for R&D/validation only. `trt-llm` returns `NotImplementedError` at startup, reserving the enum value for a follow-up PR; promoting it from conditional to required is gated on Mode B measurement in Resolve-Before-Planning Q1. Sample rate resampling (if FastCosyVoice's native rate differs from 24 kHz) happens inside the sidecar; measurement impact on TTFB budget is part of Q1.

**Startup, caching, and operator experience**

- **R12.** Model weights (~4.9 GB for `Fun-CosyVoice3-0.5B-2512`) are downloaded on first sidecar boot via `modelscope` or `huggingface_hub.snapshot_download` into a dedicated named volume **`hf_cache_fastcosy`** mounted at `/root/.cache/huggingface` inside the sidecar container (matches `hf_cache_qwen3` precedent at `infra/docker-compose.yml:209`).
- **R13.** TensorRT Flow engine artefacts are GPU-arch-specific and **cannot be baked into the image**. First sidecar boot on a new host compiles the TRT Flow engine (~2 min on L4) and caches it into a **separate named volume `trt_cache_fastcosy`** mounted at `/opt/fastcosy/trt-cache` and exposed via `FASTCOSY_TRT_CACHE_DIR` env var (if the upstream fork doesn't honor the env, a local patch in `infra/fastcosy-tts-wrapper/` routes the cache path — matches the "patch-in-place" discipline in `infra/qwen3-tts-wrapper/README.md`). Cache path is prefixed by a `FASTCOSY_CACHE_BUSTING_TOKEN` (defaulted to a hash of Dockerfile + model SHA) so that model or Dockerfile bumps auto-invalidate stale caches. `docker-compose.yml` sets `healthcheck.start_period` at **600 s** on the sidecar service (covers TRT compile + model load + first-call warmup on cold boots), matching the precedent set in `docs/solutions/tts-selection/silero-spike-findings.md`.
- **R14.** The sidecar exposes a `GET /health/ready` endpoint that returns 200 only after model load + TRT export complete. `/health/live` returns 200 as soon as the process is up, so Docker's liveness probe does not kill a container that is still compiling.
- **R15.** FastCosyVoice is **clone-only** (unlike Qwen3, which also accepts OpenAI whitelist voices like `alloy`, `echo`). When the `fastcosy` branch dispatches with a voice string that does not start with `clone:`, the factory substitutes `clone:fastcosy_default` and emits a `loguru.warning`. Mechanical pattern (substitute-and-warn) mirrors `tts_factory.py`'s Qwen3 substitution, but the eligibility policy is stricter — no OpenAI whitelist pass-through.

**Reliability and fallback**

- **R20.** Agent-side HTTP client timeout on `OpenAITTSService.audio.speech.create` is bounded to **2 s** (end-to-end, including first-chunk wait). On timeout, `ReadTimeout` is redacted to `ErrorFrame` — matching `qwen3_tts.py`'s per-utterance network-error redaction (not a runtime engine swap). A per-utterance timeout does not swap engines mid-session; it surfaces a single failed utterance to the caller. **Cross-session engine swap** is R23's concern (agent startup only), not R20's.
- **R21.** Sidecar enforces **single in-flight request**: a second concurrent request while one is synthesizing receives HTTP 503 with `Retry-After`. Agent-side handling: the `run_tts` override translates `httpx.HTTPStatusError(503)` into `ErrorFrame("TTS busy, retry")` with a caller-observable error code. **Single-LiveKit-room-at-a-time is the documented MVP invariant** (Scope Boundaries); under that invariant, 503 is a rare operator-misconfiguration signal, not a normal-traffic response. Shadow-path test for 503 is added to R17 enumeration.
- **R22.** Sidecar healthcheck considers the process unhealthy when both: (a) `/health/synth` canary ping fails (direct liveness signal), AND (b) the canary path cannot acquire the single-inference slot within `FASTCOSY_CANARY_BUDGET_S` (defaults in Deferred-to-Planning). The canary runs on a **bypass path** separate from R21's queue — it uses a fixed ~1-word utterance so its jitter contribution is bounded, and a live long-form synth in progress does not cause a false-unhealthy. The idle-window heuristic from pass-1 has been dropped — low-traffic MVP deployments cannot reliably produce the "no successful inference" signal.
- **R23.** When `fastcosy` is reachable via **either** `TTS_ENGINE=fastcosy` at startup OR `overrides.tts_engine="fastcosy"` per-session (because the engine is in `_preload_engines`), the agent's factory branch performs a **polling `GET /health/ready`** on first dispatch with a total budget matching `docker-compose.yml`'s `healthcheck.start_period` (600 s). On persistent failure: the factory drops `fastcosy` from the runtime `_preload_engines` set (matching the XTTS-ENOSPC precedent at `services/agent/main.py:451`), so subsequent dispatches return 409 rather than silently swapping voices. `CRITICAL` is logged **and** a `fastcosy_unavailable` counter increments (ops-surface metric — not silent). An optional `FASTCOSY_FALLBACK_ENGINE` (default `none`; accepts `none | piper | silero | qwen3` excluding `fastcosy` and `xtts`) swaps the selected engine for that session only, with a `fastcosy_fallback_active{to=<engine>}` counter — but default is **fail-closed** (ErrorFrame), because a silent voice swap is a product-identity regression, not graceful degradation. Piper is available as a fallback engine only when Piper itself loaded successfully at agent startup; otherwise the fallback fails closed too (no cascade).

**Safety, testing, and documentation**

- **R16.** A post-build smoke test in the sidecar `Dockerfile` **runtime stage** (after `RUN uv sync`, in the final image — not the builder / intermediate layers) executes `python -c "from fastcosyvoice import FastCosyVoice3; print('ok')"`. Matches the remediation documented in `docs/solutions/build-errors/transformers-v5-isin-mps-friendly-coqui-tts.md`.
- **R17.** `services/agent/tests/test_tts_factory.py` is extended with a new `fastcosy` branch test (dispatches the factory, mocks the sidecar HTTP endpoint). `services/agent/tests/test_engine_parity.py` **requires no edits** — it checks the frozenset/Literal consistency asserted in R1, and passes automatically once all four declaration sites are updated. Shadow-path coverage mirrors `services/agent/tests/test_qwen3_tts.py`: 5xx status, `httpx.ConnectError`, `httpx.RemoteProtocolError`, empty-body chunks, `clone:<missing-profile>`, and the factory-side substitution warning path. CI continues to run on CPU; tests mock the sidecar HTTP endpoint, they do not load the model.
- **R18.** A solutions doc lands in `docs/solutions/tts-selection/fastcosyvoice-integration.md` recording: the cross-lingual RU workaround, Python-3.10 runtime pin rationale, CUDA-12.8-vs-13.2 split, TRT Flow first-boot cache timing, solo-maintainer risk, and the CV3 same-language-clone bug. Follows the YAML-frontmatter + `problem_type` convention documented in `CLAUDE.md`.
- **R19.** Module docstring in `services/agent/fastcosy_tts.py` records Apache 2.0 licensing of both code and weights, the sidecar-architecture decision, the Python-3.12-incompat reason, the CUDA-version asymmetry rationale (with a pointer to the Q4 resolution doc), and a one-line pointer to the solutions doc in R18 — mirroring the `xtts_tts.py` module docstring's disclosure pattern.

## Architecture Diagram

```
┌────────────────────────────────────────┐
│      services/agent (Python 3.12)      │
│   CUDA 13.2 base • torch 2.11 pin      │
│                                        │
│   tts_factory.py ──► fastcosy_tts.py   │
│                       │                │
│                       │ (OpenAI-compat HTTP, streaming)
│                       ▼                │
└───────────────────────┼────────────────┘
                        │
                        │  :8002 (agent-internal net)
                        │
┌───────────────────────▼────────────────┐
│  infra/fastcosy-tts-wrapper (Py 3.10)  │
│   CUDA 12.8 base • torch 2.7.1+        │
│                                        │
│   FastAPI  ──►  FastCosyVoice3         │
│      │              │                  │
│      │              │ TRT-Flow + compile
│      │              │ auto_stress (silero-stress built-in)
│      │              ▼                  │
│      │           Fun-CosyVoice3-0.5B   │
│      │           (llm.rl.pt)           │
│      ▼                                 │
│  /opt/voice_library/profiles/          │
│     fastcosy_default/                  │
│       meta.json  <ref_audio_filename>  │
│     (shared layout with XTTS, Qwen3;   │
│      filename looked up from meta.json)│
│                                        │
│   hf_cache_fastcosy (volume)           │
│     • HF weights (~4.9 GB)             │
│   trt_cache_fastcosy (volume)          │
│     • TRT Flow engine (built on        │
│       first boot, GPU-arch-specific)   │
└────────────────────────────────────────┘
```

## Success Criteria

- **Functional.** `TTS_ENGINE=fastcosy` environment flip starts an agent that dispatches live Russian audio through a LiveKit room end-to-end. Per-session override via `overrides.tts_engine = "fastcosy"` routes a single session without affecting others.
- **Latency.** Measured TTFB on L4 GPU (AWS `g6.xlarge` or equivalent), Mode B, over 10 Russian utterances (5 short conversational ≤ 8 words, 5 long narrative ≥ 30 words): **p50 ≤ 600 ms, p95 ≤ 800 ms** from EOS to first audio frame landed on the LiveKit track. **Kill-switch**: if measured p50 > 750 ms or p95 > 1000 ms on L4 Mode B, v1 is **deferred** — not escalated to Mode C. The escalation to Mode C is a separate decision made after the managed-API alternatives comparison (Resolve-Before-Planning Q5) has been re-evaluated with measured data in hand.
- **Quality.** R21 blind-listen (per `docs/solutions/tts-selection/r21-protocol.md`): FastCosyVoice is **strictly preferred to Piper on Russian** — pass criterion ≥ 2 of 3 raters prefer FastCosyVoice > Piper on ≥ **7 of 10** utterances at 24 kHz int16 mono, **OR** absolute rater MOS ≥ 4.0 on ≥ 6 of 10 utterances. Parity-with-Piper is not sufficient: a 4.9 GB GPU engine with sidecar complexity must earn its keep on perceivable quality.
- **Barge-in.** Cancel-latency regression on interrupted Russian synthesis is **within 300 ms of the Piper R11b baseline** for utterances ≥ 20 words, measured at a 500 ms mid-synth interrupt. The barge-in rule from `docs/solutions/tts-selection/silero-spike-findings.md` applies.
- **No regression.** The existing Piper / Silero / Qwen3 / XTTS parity tests (`test_engine_parity.py`, `test_tts_factory.py`) pass with the new branch added. Existing env var precedence semantics for `TTS_VOICE` and `XTTS_TTS_VOICE` are preserved.
- **Build reproducibility.** Warm-cache: `docker compose up -d --build` reaches `healthy` on the fastcosy service within **600 s** (HF weights already in `hf_cache_fastcosy` volume). Cold-download: first-ever boot on a clean host (~4.9 GB weight pull + TRT Flow compile) completes within **1800 s**. Both budgets documented in the sidecar README so contributors with slow egress don't misdiagnose a stuck build.

## Scope Boundaries

- **Not in v1**: Mode C (TensorRT-LLM fast path, 0.47 s TTFB target). `trt-llm` is a reserved env value only — enabling it is a follow-up PR gated on measured Mode-B data.
- **Not in v1**: voice cloning of a specific Russian speaker with a Russian reference audio. Blocked on [`FunAudioLLM/CosyVoice#1790`](https://github.com/FunAudioLLM/CosyVoice/issues/1790) which is closed-as-stale without fix; cross-lingual (Chinese ref + Russian target) is the only supported path.
- **Not in v1**: retiring or gating any existing engine. Piper remains the default `TTS_ENGINE`. Silero, Qwen3, XTTS stay selectable with their current semantics.
- **Not in v1**: UI changes beyond the existing three-button engine selector in `apps/web`. If operators want to expose `fastcosy` in the UI, that's a follow-up PR.
- **Not in v1**: a dedicated `bench_tts` multi-engine scorecard harness. The v1 acceptance measurement is scripted manually for `fastcosy` against the existing `r21-protocol.md` and R11b baselines; the generalized harness is a separate ideation survivor.
- **Not in v1**: Russia-native managed APIs (Yandex SpeechKit, SaluteSpeech). Distinct work track.
- **Not in v1**: multi-session concurrency tuning. Sidecar defaults to a single inference worker; load-testing multi-session behavior is scheduled only if the single-tenant MVP demand grows.

## Key Decisions

- **Sidecar, not in-process.** FastCosyVoice pins Python `>=3.10,<3.12` in its `pyproject.toml` and CUDA 12.8 wheels in its `[tool.uv.sources]`. The agent runs Python 3.12 + CUDA 13.2. In-process integration would either require patching the fork's pin (owns-a-fork-of-a-fork compounding maintenance) or regressing the agent's Python/CUDA versions (retests all four existing engines + gigaam). The Qwen3-TTS pattern already vendored under `infra/qwen3-tts-wrapper/` proves the shape works and gives us a matching Dockerfile/OpenAI-compat/voice-library template.
- **Adapter shape: OpenAITTSService subclass, ~85 LOC `run_tts` override.** Matches `services/agent/qwen3_tts.py` — **not** a "thin error-redaction subclass." Streaming mode is measurement-gated (Q1); today's qwen3_tts.py ships non-streaming on L4 because RTF ~3x caused browser buffer underruns. Don't claim a streaming precedent that the codebase already rejected on the same hardware.
- **Mode B (TRT-Flow + torch.compile), not Mode C — conditionally.** Mode C's TensorRT-LLM adds ~1.5–2 GB to the sidecar image, couples strictly to CUDA ABI, and doubles build time. On Ada L4, the B-vs-C gap is likely smaller than the 330 ms observed on Ampere — but this is extrapolation, not measurement. If Q1 measures Mode B above kill-switch, the decision is **not** automatic-escalate to Mode C; it is re-evaluate against managed-API alternatives first.
- **Vendored, not submoduled.** Matches `infra/qwen3-tts-wrapper/README.md#Why-vendored-not-submoduled`. FastCosyVoice is solo-maintained (1 author, 69 ⭐, last push 2026-03-07).
- **`llm.rl.pt` default over `llm.pt`.** 3.79 % RU WER vs 6.77 % (CV3 paper Table 5). Same Apache 2.0 license, same size.
- **Cross-lingual-only voice strategy.** Upstream issue #1790 is closed-as-stale; RU-reference clone is not supported. Chinese-reference workaround is the shipped path; operators who want Russian-speaker cloning are pointed at Qwen3 or XTTS. **This means criterion (d) from Problem Frame — "zero-shot voice control" — is satisfied only for generic-voice output, not speaker-specific cloning in v1.**

## Portfolio Posture

- **Engine-count cap — XTTS retirement is a committed rule, not an evaluation.** If FastCosyVoice passes v1 gates (Q1 + Q2a + Q2b or Q2b-defer variant) **and** R21 measures FastCosyVoice ≥ XTTS on RU quality at strict preference (≥ 2 of 3 raters prefer FastCosyVoice > XTTS on ≥ 6 of 10 utterances), **XTTS is retired in the same PR** (or a blocking follow-up merged within 2 weeks). XTTS carries a CPML commercial blocker; keeping it alongside a cleaner Apache-2.0 equivalent is compounding technical debt. If R21 does not demonstrate FastCosyVoice ≥ XTTS on RU quality, XTTS stays — but that is itself a signal that the ship is marginal. Keep the count of vendored-solo-maintained upstreams from compounding: adding FastCosyVoice brings it to 2 (alongside Qwen3 wrapper); any future third such upstream requires explicit portfolio re-evaluation, not incremental acceptance.
- **Managed-API alternatives evaluated.** Cartesia Sonic-3 (~90 ms TTFB managed), ElevenLabs Flash v2.5 (~75 ms), and SaluteSpeech (native RU, 200 K chars/mo free tier) all ship RU and all beat self-hosted latency. Selection of FastCosyVoice over these rests on: (1) **offline-capable / no-egress-cost operation**, (2) **full model control for future fine-tuning**, (3) **voice-identity ownership** not dependent on vendor API continuation. If any of these is not a hard requirement for the product, the managed path is cheaper. **Resolve-Before-Planning Q5 forces this comparison to be on paper before scaffolding begins.**

## Dependencies / Assumptions

- **AWS `g6` family with L4 GPU** is the deployment target. Host driver R580+ *nominally* satisfies both CUDA 13.2 (agent) and CUDA 12.8 (sidecar) — the CUDA runtime uses forward-compatibility with a newer driver. **Not tested in this repo**; see Resolve-Before-Planning Q4.
- **CosyVoice 3 RU cross-lingual output quality is not independently benchmarked.** CV3 paper reports WER (pronunciation accuracy), not MOS (naturalness). R21 acceptance bar is the only gate.
- **Solo-maintainer vendored-upstream count = 2** after this ships (Qwen3 wrapper + FastCosyVoice). Acceptance is bounded by the Portfolio Posture cap, not open-ended.
- **Apache 2.0 license is a point-in-time assertion.** FastCosyVoice (code) and `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (weights) are both Apache 2.0 as of 2026-04-23. Retroactive license flips have happened in this space (Fish-Speech, March 2026). Provenance entry (R2) includes a SHA256 of the LICENSE file for drift detection; license re-check is part of the vendored-upstream re-sync PR cadence.

## Outstanding Questions

### Resolve Before Planning

- **Q1. [Affects Latency SC + R3 + R9][Needs measurement]** Spike on two independent `g6.xlarge` instances (each ≥ 1 hour, ≥ 50 RU utterances split short/long per R21 protocol). Discard the first 5 iterations as torch.compile warmup. Record p50/p95 TTFB + RTF + native output sample rate + warm-vs-cold distribution. Gate: **across both runs**, p50 ≤ 750 ms and p95 ≤ 1000 ms AND the two runs' p95s are within 15% of each other (variance band). If either bound fails or variance is wider than 15%, v1 is deferred (not auto-escalated to Mode C — re-evaluate against managed-API alternatives first). If RTF ≥ 1.0x, R3 ships non-streaming by default.
- **Q2a. [Affects R6 + Quality SC][Needs measurement] Engine viability.** Run R21 blind-listen on **one representative Chinese reference clip** (starting with FastCosyVoice `asset/zero_shot_prompt.wav`). Pass criterion: output is intelligibly Russian + streams cleanly at Q1-measured TTFB. If engine cannot produce R21-passable RU on any clip, v1 is deferred (engine issue, not voice-choice issue).
- **Q2b. [Affects R6][Needs measurement] Default voice selection.** If Q2a passes, audition ≤ 3 candidate Chinese clips (including `refs/019.wav` and FastCosyVoice's `asset/zero_shot_prompt.wav`) against the strict-preference-over-Piper form (≥ 2 of 3 raters prefer > Piper on ≥ 7 of 10 utterances, OR absolute MOS ≥ 4.0 on ≥ 6 of 10). Ship the winner as `fastcosy_default`. If none clears the strict gate but Q2a passes, v1 still ships — operators supply their own ref clip; `fastcosy_default` becomes a "functional but unrecommended" baseline with a loguru warning on first use.
- **Q3. [Affects R7][Needs measurement]** Run ~100 representative LLM-output RU sentences (from agent logs — include EN/RU code-switching, numerical expressions, already-stress-marked text, punctuation edge cases) through `auto_stress`. Verify no crash + sanity-check a sampled subset. If unsafe, default flips to `FASTCOSY_AUTO_STRESS=false`.
- **Q4. [Affects Dependencies + R2][Needs measurement]** Verify on an actual `g6.xlarge` instance with R580+ driver that **all three GPU consumers** coexist — vLLM (LLM, per CLAUDE.md architecture) + agent (GigaAM STT + preloaded TTS engines) + fastcosy sidecar (TRT Flow) — running a concurrent workload for 10 minutes without OOM, with measured VRAM headroom ≥ 2 GB. Driver/toolkit pass criterion: `torch.cuda.is_available()` + a non-trivial matmul in each container, peak VRAM under 22 GB (2 GB reserved). Pin the specific `nvidia-container-toolkit` version and record it in `infra/` along with a tolerance band (patch versions accepted, minor versions require re-verification).
- **Q5. [Affects Portfolio Posture][Decision]** One-page comparison of FastCosyVoice-sidecar vs SaluteSpeech (200 K chars/mo free + native RU), Cartesia Sonic-3 (~90 ms), ElevenLabs Flash v2.5 (~75 ms). Columns: measured RU TTFB, commercial license, RU quality signal, cost at projected load, data-residency, maintenance surface. Document the specific requirement that selects self-host (offline/airgap, fine-tuning ownership, or voice-identity independence) — if no such requirement exists, the managed path is cheaper and the fastcosy project doesn't justify its complexity.

### Deferred to Planning

- **[Affects R12, R13][Technical]** Base image for the sidecar: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` vs `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. Planning picks the smaller image that satisfies TensorRT wheel compatibility.
- **[Affects R13][Technical]** TRT Flow engine caching strategy: first-boot-cache-to-named-volume (this spec) vs bake-per-arch-into-image (CI produces L4-specific variant).
- **[Affects R13][Technical]** TRT compile-failure recovery: on compile failure, sidecar deletes partial cache artifact, logs failure with CUDA/TensorRT/driver/compute-capability, fails `/health/ready` with structured error. `FASTCOSY_CACHE_BUSTING_TOKEN` (defaulted to a hash of Dockerfile + model SHA) prepended to cache path so version bumps auto-invalidate.
- **[Affects R2][Technical]** OpenAI-compat server: write from scratch in `infra/fastcosy-tts-wrapper/api/` or port Qwen3 wrapper's server layer as skeleton. Choice depends on diff size.
- **[Affects R17][Technical]** Exact pytest fixture for mocking the sidecar HTTP contract. `tests/test_qwen3_tts.py` likely has a reusable fixture.
- **[Affects R9][Needs research]** FastCosyVoice's sidecar-level deps (`transformers>=4.53.1` no-upper-bound will resolve to 5.x) — dry-run `uv lock` inside `infra/fastcosy-tts-wrapper/` and record the resolved `transformers` version. If it resolves to 5.x, run the sidecar's import path end-to-end; pin sidecar-local `transformers<5.0` if any Tortoise-family or deprecated-symbol import fails (same class as the coqui-tts incident in `docs/solutions/build-errors/`).
- **[Affects R1 + AgentConfig][Technical]** Add `fastcosy_base_url`, `fastcosy_api_key`, `fastcosy_tts_voice` fields to `AgentConfig` following the `qwen3_*` precedent; update `pipeline.py::AgentConfig.__post_init__` validation.
- **[Affects R23][Technical]** `FASTCOSY_FALLBACK_ENGINE` default — `piper` (fail-open, recommended for live sessions) vs `none` (fail-closed, preferred when operators expect loud failure). Set repo default here before planning starts.
- **[Affects R21, R22][Technical]** Exact `Retry-After` seconds on the 503-on-busy response; exact `FASTCOSY_IDLE_HEALTH_WINDOW` value; `/health/synth` implementation (canary utterance, timeout).

## Next Steps

`-> Resume /ce-brainstorm` to resolve the **five Resolve-Before-Planning questions** (L4 latency spike, R21 clip audition, `auto_stress` corpus probe, CUDA cohabitation verification, managed-API alternatives comparison) before implementation planning begins. Each gate can cause v1 to be deferred; running `/ce-plan` before they resolve would invent implementation steps around risks whose disposition is unknown.

Once Q1–Q5 resolve with pass results, `-> /ce-plan` for structured implementation planning. Expected first phase: sidecar scaffolding (copy Qwen3 wrapper skeleton → replace with FastCosyVoice's `FastCosyVoice3` class → wire OpenAI-compat API + reliability requirements R20–R23), then Dockerfile + TRT Flow export wiring, then agent-side factory branch + four whitelist-sync sites + module docstring + tests, then live L4 measurement confirming Success Criteria.

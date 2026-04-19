# Phase A results — STT A/B experiment

**Status:** ambiguous — see "Decision" at bottom.

**Last updated:** 2026-04-19 (Units 1-3 complete)

## Run metadata

- **Commit SHA:** 9996495 (feat/stt-ab-experiment)
- **Docker image:** `agent-gigaam-test:latest` on `project-800ms-instance` (18.8 GB)
- **GPU:** L4 24 GB, us-central1-a
- **Eval set size:** 70 utterances from `bond005/sberdevices_golos_10h_crowd`, test split
- **Eval set characteristics:** Russian voice-assistant commands, 1-10s each, 16 kHz mono
- **Filter thresholds in play (GigaAM):**
  - `min_duration_seconds`: 0.30 (default — not tuned)
  - `min_token_count`: 2 (default — not tuned)

## Dataset selection notes

Originally targeted `mozilla-foundation/common_voice_17_0` but Mozilla pulled CV from HF in Oct 2025 (now behind Mozilla Data Collective signup). Tried `google/fleurs` next — blocked by HF `datasets` v3+ dropping loading-script support. Settled on `bond005/sberdevices_golos_10h_crowd`: non-gated, parquet-native, Russian voice-assistant register.

**⚠ Known bias:** Golos is Sberdevices' own dataset. GigaAM-v3 is Sber's model, near-certainly trained on or near this data. Treat absolute GigaAM numbers on Golos as a **home-court ceiling**, not an unbiased estimate.

## VRAM verification (Unit 1 gate)

**Run:** 2026-04-19 on `project-800ms-instance` (us-central1-a, L4 24 GB), image `agent-gigaam-test:latest` (18.8 GB on disk). L4 started empty (baseline 2.2 GB is NVIDIA CUDA container overhead + system).

| Component | Peak MiB | Peak GB | Δ from previous |
|---|---|---|---|
| Baseline (before STT loads) | 2,292 | 2.24 | — |
| + Whisper `large-v3` int8_float16 (25.7s load) | 4,307 | 4.21 | +2,015 MiB |
| + GigaAM-v3 CTC (4.7s load) | 4,749 | 4.64 | +442 MiB |
| + Silero VAD (0.1s load, CPU path) | 4,749 | 4.64 | +0 MiB |
| **Total STT+VAD footprint** | **2,457 MiB** | **2.40 GB** | — |
| **Peak resident** | **4,749 MiB** | **4.64 GB** | — |
| **Ceiling** | 22,528 MiB | 22.0 GB | — |
| **Verdict** | ✅ PASS | | peak is 21% of ceiling |

**Observations:**
- GigaAM-v3 is ~2× smaller than the brainstorm's ~1 GB estimate (442 MiB actual).
- Whisper-large-v3 at `int8_float16` is ~1 GB smaller than estimated (2 GB vs 3 GB).
- Startup time of both models combined: ~30s. Fast enough that agent restart isn't a deployment pain.
- Plenty of headroom — vLLM (Qwen-7B-AWQ, ~13 GB) could comfortably co-reside if we ever decide to move LLM back on-box.

## Per-stack WER (Unit 3)

With text normalization (lowercase, strip punctuation, expand 0-20 digits to Russian words):

| Stack | Mean WER | n | Reject rate |
|---|---|---|---|
| Whisper | 16.58% | 70 | 0 (no filter thresholds exercised) |
| GigaAM | 2.81% | 70 | 0 |

**Δ WER (whisper − gigaam):** +13.77 points (positive = GigaAM wins)

**Without normalization, raw numbers:** Whisper 68.02% / GigaAM 2.81% / Δ = +65.21. The 51-point normalization delta isolates how much of the raw gap was stylistic noise (case/punct/digit style) vs real model quality.

## Per-category breakdown

| Category | Whisper WER | GigaAM WER | Δ | Notes |
|---|---|---|---|---|
| `clean` | 15.83% | 2.94% | +12.89 | n=67, Golos voice-assistant commands |
| `short` (<2s) | 33.33% | 0.00% | +33.33 | n=3, too few samples to trust — hits Whisper's known short-audio weakness |
| `noisy` | — | — | — | not represented in Golos crowd split |
| `game_specific` | — | — | — | not represented in Golos crowd split |
| `stress` | — | — | — | not represented in Golos crowd split |

## Where the 13.8-pt gap comes from

Qualitative breakdown of Whisper's residual errors (from inspecting rows with WER > 0.3 in the CSV):

| Error type | Est. WER contribution | Real or artifact? |
|---|---|---|
| Multi-digit numbers (`28` vs `двадцать восьмое`) not covered by 0-20 normalizer | ~4 pts | Artifact — would need `num2words` to be fair |
| Latin transliteration of Cyrillic abbreviations (`fgbnu`, `U2`, `Dixieland`) | ~3 pts | Mixed — Whisper's output is readable; GigaAM matches truth because of Sber training |
| Case/grammar inflection (`Вячеслава` vs `вячеславу`) | ~2 pts | Real but minor — Whisper picks nominative, truth is dative |
| Rare proper nouns (Tatar/Chechen names, brand names) | ~3 pts | Real Whisper weakness |
| One classic Whisper hallucination (`Субтитры сделал DimaTorzok` on a short clip) | ~1.5 pts | Real Whisper failure |

**Adjusted for normalization fairness, the model-quality gap is ~5-6 pts in GigaAM's favor on this register.**

## Latency (from harness timing)

Approximate, from the per-utterance `_ms` columns in the CSV:

| Stack | Mean ms / utterance | Notes |
|---|---|---|
| Whisper | ~370 ms | faster-whisper large-v3, int8_float16, CUDA |
| GigaAM | ~330 ms | gigaam v3_ctc, torch, CUDA, first call includes ffmpeg subprocess |

Latency is essentially tied — neither stack's latency is a decision driver for Phase A.

## Decision

**Outcome:** ambiguous-with-caveats — formally meets the 5-pt "decisive" gate but on a test with known bias.

**Reasoning:**
- The 13.8-pt raw Δ and ~5-6-pt normalization-adjusted Δ both exceed the plan's 5-pt decisive gate.
- **However, the test has known contamination bias**: Golos is Sber's own dataset, and GigaAM-v3 was almost certainly trained on or near it. GigaAM's 2.8% WER likely includes memorization, not pure generalization. Real out-of-distribution WER for GigaAM is likely 5-10%.
- Whisper's 16.6% WER is in-line with published Russian ASR benchmarks for short-command audio (its known weak register). On long-form/conversational audio, Whisper typically performs much better.
- The question Phase A was supposed to answer — "is there an obvious gap" — got a yes, but the test doesn't rule out the answer being "yes, because we picked the test GigaAM was trained on."

**Next step options (user to pick):**

1. **Accept decisive result, skip Phase B.** Adopt GigaAM as the default on the voice-command/game-NPC register (which matches our product target). Keep Whisper for one release as hedge. Proceed to Unit 9 cleanup-for-winner. Fastest.
2. **Validate on an unseen register first.** Re-run WER smoke on `bond005/rulibrispeech` (Russian audiobook reads — NOT Sber's data, likely out-of-distribution for both models). If gap persists ≥5 pts → confirm decisive GigaAM. If gap shrinks → accept Phase B is needed. Adds ~30 min.
3. **Proceed to Phase B anyway.** Blinded subjective A/B on real game audio (the actual target domain). Strongest evidence but costs days. Appropriate if we don't trust any offline test enough to skip user judgment.

**Recommended:** Option 2 — cheap additional evidence that cleanly resolves the contamination question before committing to a week of Phase B or rolling GigaAM out.

## Artifacts

- Per-utterance CSV: `services/agent/eval/results/2026-04-19-wer-smoke.csv` (committed)
- Eval set audio + truth: `services/agent/eval/russian_eval_set/` (70 WAV + truth.jsonl, committed)
- `verify_vram.py` terminal log: captured in the VRAM section above (Unit 1)

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

## OOD validation — rulibrispeech (2026-04-19 follow-up)

Ran a second WER smoke against `bond005/rulibrispeech` (Pushkin and other 19th-c. Russian literary audiobook reads, validation split, 70 utterances). Out of distribution for both stacks: it is audiobook-register literary Russian, not Sber voice-assistant data and not dominant Whisper training content.

| Stack | Mean WER | n | Δ |
|---|---|---|---|
| Whisper | 10.07% | 70 | — |
| GigaAM | 6.02% | 70 | +4.06 pts (GigaAM wins) |

**This falls under the 5-pt decisive gate — formally inconclusive per the plan.**

### Register shift analysis

| Register | Whisper | GigaAM | Δ |
|---|---|---|---|
| Voice commands (Golos — Sber home court) | 16.58% | 2.81% | +13.77 (inflated by contamination) |
| Audiobook literary Russian (rulibrispeech OOD) | 10.07% | 6.02% | +4.06 (clean) |

~9.7 pts of the Golos gap was Sber home-court advantage, as suspected. The clean OOD gap of ~4 pts is the real model-quality delta in GigaAM's favor on Russian.

### OOD error character

Both models struggle on archaic/rare Russian vocabulary, with different failure modes:

- **Whisper:** word-boundary loss (`остолбенев, бледнея` → `о столбине в бледне`), phonetic mis-spellings on archaic words (`млат` → `млад`, `булат` → `булад`), archaic `лет` (flight) → modern `лед` (ice).
- **GigaAM:** "reasonable but wrong" substitutions that modernize archaic forms (`младых` → `молодых`, `главы` → `головы`, `клобук` → `клобок`).

Neither is broken on OOD. Whisper's errors tend to be more egregious (nonsense tokens); GigaAM's tend to be more plausible-but-wrong (silent modernization).

## Decision

**Outcome:** inconclusive per the plan's 5-pt gate, but with a clean real edge of ~4 pts for GigaAM on OOD Russian plus a larger edge on voice-command register (even after contamination adjustment).

**Reasoning:**
- Golos result alone can't justify skipping Phase B — home-court contamination is real and accounts for ~10 pts of the raw gap.
- OOD result is cleaner evidence and shows GigaAM at 6.0% vs Whisper at 10.1% — a meaningful but sub-decisive real edge.
- For the target use case (Russian voice game NPCs, contemporary spoken Russian, short utterances), the relevant register is closer to Golos than to Pushkin. Expected production gap: 5-10 pts.
- Both stacks clear the "good enough" bar (both are under 20% WER on both tests).

**Next step options:**

1. **Adopt GigaAM as default; skip the full Phase B UX work.** Ship GigaAM as the configured default with the existing `GigaAMSTTService` wrapper and startup preload (Units 1-2). Keep Whisper in code behind the existing kill-switch env var (`DISABLE_STACK_GIGAAM=false` / flip to `true` to fall back) for one release. Skip Units 4-8 (ratings migration, SPA dual-button, blinded A/B). Rationale: the 4-pt OOD edge plus Sber register fit plus our Cyrillic-only text pipeline's match to GigaAM's output style is enough evidence for a low-risk swap with a fast rollback. If production metrics regress, flip the env var.
2. **Execute Phase B in full.** Units 4-8. Blinded subjective A/B on real game audio. Strongest evidence, costs several days.
3. **Execute Phase B partial (Units 4-6 only).** Build the ratings table + API endpoints + agent stack routing + Prometheus stack-labeled latency histogram — these have value beyond this experiment (observability, rating-feedback pipeline). Skip the dual-button SPA (Unit 7) and n≥30 collection (Unit 8). Use production metrics (error rates, latency) as the decision signal rather than blinded subjective ratings.

**Recommended:** Option 1. The evidence is good enough to swap with a kill-switch hedge, and Phase B's cost-to-benefit is poor given we already have a 4-pt OOD edge plus register fit. Option 3 is a reasonable compromise if you want production observability on the swap.

## Artifacts

- Per-utterance CSV: `services/agent/eval/results/2026-04-19-wer-smoke.csv` (committed)
- Eval set audio + truth: `services/agent/eval/russian_eval_set/` (70 WAV + truth.jsonl, committed)
- `verify_vram.py` terminal log: captured in the VRAM section above (Unit 1)

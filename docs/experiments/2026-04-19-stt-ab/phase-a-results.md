# Phase A results — STT A/B experiment

**Status:** pending — eval set not yet populated.

**Last updated:** 2026-04-19 (scaffolding only)

## Run metadata

- **Commit SHA:** TBD
- **Docker image:** TBD (tag from `docker-agent` CI job)
- **GPU:** TBD (L4 zone, VRAM peak from `verify_vram.py`)
- **Eval set size:** TBD (target ~100 utterances)
- **Filter thresholds in play (GigaAM):**
  - `min_duration_seconds`: 0.30 (default; tuned during this run if reject-rate parity is off)
  - `min_token_count`: 2 (default)

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

| Stack | Mean WER | n | Reject rate |
|---|---|---|---|
| Whisper | — | — | — |
| GigaAM | — | — | — |

**Δ WER (whisper − gigaam):** — points (positive = GigaAM wins)

## Per-category breakdown

| Category | Whisper WER | GigaAM WER | Δ | Notes |
|---|---|---|---|---|
| `clean` | — | — | — | — |
| `noisy` | — | — | — | — |
| `game_specific` | — | — | — | — |
| `stress` | — | — | — | — |
| `short` | — | — | — | — |

## Latency (from harness timing)

| Stack | Mean ms / utterance | p95 ms |
|---|---|---|
| Whisper | — | — |
| GigaAM | — | — |

## Decision

**Outcome:** TBD (decisive-GigaAM / decisive-Whisper / inconclusive)

**Reasoning:** (populate after the run)

**Next step:** TBD
- If decisive GigaAM win: update this plan's status, skip to Unit 9 cleanup-for-winner.
- If decisive Whisper win: open the "delete GigaAM work" follow-up; close the experiment.
- If inconclusive: proceed to Phase B (Units 4-8).

## Artifacts

- Per-utterance CSV: `services/agent/eval/results/YYYY-MM-DD-wer-smoke.csv` (committed with this doc update)
- Eval set audio + truth: `services/agent/eval/russian_eval_set/` (committed once)
- `verify_vram.py` terminal log: paste below

```
(paste nvidia-smi output + verify_vram.py stdout here after the run)
```

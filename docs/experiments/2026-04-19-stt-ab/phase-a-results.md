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

| Component | Peak MiB | Peak GB |
|---|---|---|
| Baseline (before STT loads) | — | — |
| + Whisper `large-v3` int8_float16 | — | — |
| + GigaAM-v3 CTC | — | — |
| + Silero VAD | — | — |
| **Total** | — | — |
| **Ceiling** | 22528 | 22.0 |
| **Verdict** | — | — |

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

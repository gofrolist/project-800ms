# Russian eval set — STT A/B experiment (Unit 3)

~100 utterances for offline WER comparison of Faster-Whisper vs GigaAM-v3.

## Layout

    russian_eval_set/
    ├── README.md           (this file)
    ├── truth.jsonl         (one JSON per line; see schema below)
    ├── clip-NNN.wav        (16 kHz mono; NNN = 001..100)
    └── ...

## truth.jsonl schema

```jsonl
{"audio": "clip-001.wav", "text": "привет как дела", "category": "clean"}
{"audio": "clip-002.wav", "text": "я хочу купить меч", "category": "game_specific"}
```

Required keys:
- `audio`: filename relative to this directory
- `text`: ground-truth transcript (Russian, no punctuation, lowercase preferred for WER stability)
- `category`: one of `clean`, `noisy`, `game_specific`, `stress`, `short`

## Target composition (~100 utterances)

| Category | Count | What it tests |
|---|---|---|
| `clean` | ~30 | Studio-quality speech, game-adjacent vocabulary. Baseline WER. |
| `noisy` | ~20 | Background music / room ambiance. Robustness. |
| `game_specific` | ~25 | Character names, items, quest keywords — terms neither model has seen at training. Domain adaptation. |
| `stress` | ~15 | Stress-mark-sensitive pairs (замок/замок, мука/мука). Russian phonology. |
| `short` | ~10 | < 2s utterances. Tests the min-duration filter's parity. |

## Sourcing

Options, in priority order:

1. **Mozilla Common Voice Russian** — free, labeled, large. Good for `clean` and `noisy` buckets. Download the ru-RU locale, sample ~50 utterances at random, manually truth-check.
2. **Self-recorded / internal** — required for `game_specific` (nobody has recorded lore keywords from our domain). ~25 utterances of a native speaker reading curated phrases.
3. **Minds-14 RU / NEMO public Russian sets** — if the Common Voice coverage is thin.

Record at 16 kHz mono WAV. If sourcing at a different sample rate, resample before committing.

## Running the harness

```bash
# From the agent Docker container on the L4 host:
python scripts/wer_smoke.py
```

Results land in `services/agent/eval/results/YYYY-MM-DD-wer-smoke.csv`.

## Decision gate

The harness prints a decision at the end based on the delta:

- Δ WER ≥ 5 points (GigaAM better) → **skip Phase B**, adopt GigaAM default.
- Δ WER ≥ 5 points (Whisper better) → **stop**, delete GigaAM work.
- Δ WER < 5 points or mixed per-category → **proceed to Phase B** (blinded subjective A/B).

Write up the decision + reasoning in `docs/experiments/2026-04-19-stt-ab/phase-a-results.md` and link the committed CSV.

"""Sample N Russian utterances from Mozilla Common Voice for the eval set.

Streams `mozilla-foundation/common_voice_17_0` (ru, test split) via HF
datasets, filters by duration, resamples to 16 kHz mono, saves as
int16 WAV, appends one line per clip to truth.jsonl.

This is one-shot scaffolding — run once to populate the eval set, then
delete the script alongside the rest of the experiment's cleanup. Not
productized.

Requires (install transiently via pip — not added to pyproject):
    datasets soundfile librosa

Usage (from inside the agent container):
    HUGGING_FACE_HUB_TOKEN=... python scripts/sample_common_voice.py --out /app/eval/russian_eval_set --n 70

HF auth: Common Voice is a gated dataset on HF — you must have accepted
the license on https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
with the same HF account the token is scoped to.

Categorisation: CV doesn't expose a noisiness label; we approximate
clean vs short by duration. Human re-labeling can move a subset to
`noisy` or `stress` after listening (edit truth.jsonl by hand).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output eval dir (holds WAVs + truth.jsonl)")
    parser.add_argument("--n", type=int, default=70, help="Target number of clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default="mozilla-foundation/common_voice_17_0",
        help="HF dataset id (override for newer versions or non-gated mirrors)",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="Max CV rows to scan when sampling — caps streaming cost",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports only inside main — the script is installed transiently,
    # don't force them on any other caller that might import this file.
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415

    print(f"Streaming {args.dataset} ({args.split}, ru)...")
    ds = load_dataset(args.dataset, "ru", split=args.split, streaming=True)

    # Gather candidates with basic quality filtering. Stream stops at
    # max_scan so we don't burn the whole split on a 70-clip sample.
    candidates: list[dict] = []
    for i, row in enumerate(ds):
        if i >= args.max_scan:
            break
        audio = row.get("audio") or {}
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is None or sr is None:
            continue
        duration = len(arr) / sr
        if duration < 1.0 or duration > 10.0:
            # Too short: not enough signal. Too long: disproportionate
            # decode cost for a smoke set.
            continue
        text = (row.get("sentence") or "").strip()
        if not text:
            continue
        candidates.append({"array": arr, "sample_rate": sr, "text": text, "duration": duration})

    print(f"Scanned {i + 1} rows, {len(candidates)} passed filters.")

    random.seed(args.seed)
    random.shuffle(candidates)
    samples = candidates[: args.n]
    if len(samples) < args.n:
        print(
            f"WARNING: only {len(samples)} clips survived filters; wanted {args.n}. "
            "Consider raising --max-scan."
        )

    truth_path = out_dir / "truth.jsonl"
    # Append rather than overwrite — preserves the existing header
    # comments and lets this be run in multiple passes if desired.
    with truth_path.open("a") as f:
        for i, sample in enumerate(samples, 1):
            filename = f"cv-{i:03d}.wav"
            arr = sample["array"]
            sr = sample["sample_rate"]

            # Resample to 16 kHz mono — our Pipecat pipeline + WER harness
            # both expect this shape. CV audio is typically 48 kHz mono.
            if sr != 16_000:
                arr = librosa.resample(
                    np.asarray(arr, dtype=np.float32), orig_sr=sr, target_sr=16_000
                )
                sr = 16_000

            # float [-1, 1] → int16
            data = np.clip(np.asarray(arr) * 32767, -32768, 32767).astype(np.int16)
            sf.write(str(out_dir / filename), data, sr)

            # Bucket: short < 2s, everything else `clean`. Human can
            # reclassify to `noisy` / `stress` after listening.
            category = "short" if sample["duration"] < 2.0 else "clean"

            entry = {
                "audio": filename,
                "text": sample["text"],
                "category": category,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} clips to {out_dir}")
    print(f"Appended {len(samples)} entries to {truth_path}")
    print("\nSample of what was written:")
    for sample in samples[:5]:
        print(f"  [{sample['duration']:.1f}s] {sample['text'][:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

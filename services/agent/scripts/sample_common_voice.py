"""Sample N Russian utterances from bond005/sberdevices_golos_10h_crowd for the eval set.

Streams the dataset (test split) via HF datasets, filters by duration,
saves as 16 kHz mono int16 WAV, appends one line per clip to truth.jsonl.

Dataset history for this experiment:
  - Originally targeted Mozilla Common Voice 17 → Mozilla pulled CV from
    HF in Oct 2025, dataset page is an empty stub now.
  - Then tried google/fleurs → HF datasets v3+ dropped support for
    loading scripts, fleurs.py won't load.
  - Settled on Sber Golos (10h crowd, non-gated, parquet-native) — closer
    to voice-assistant register than FLEURS (read news) would have been
    anyway. Both Whisper and GigaAM are trained on or near Golos so we
    still get a sanity cross-check.

This is one-shot scaffolding — run once to populate the eval set, then
delete the script alongside the rest of the experiment's cleanup. Not
productized.

Usage (from inside the agent container):
    python scripts/sample_common_voice.py --out /app/eval/russian_eval_set --n 70

Categorisation: Golos crowd is already casual/command-style speech; we
mark short clips as `short` and the rest as `clean`. Phase B catches
noise/stress variance on real game audio.
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
        default="bond005/sberdevices_golos_10h_crowd",
        help="HF dataset id",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config (Golos has no configs; FLEURS used 'ru_ru', CV used 'ru')",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="Max rows to scan when sampling — caps streaming cost",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports only inside main — the script is installed transiently,
    # don't force them on any other caller that might import this file.
    import io  # noqa: PLC0415

    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from datasets import Audio, load_dataset  # noqa: PLC0415

    print(f"Streaming {args.dataset} ({args.config or '-'}, {args.split})...")
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # datasets v3 defaults to torchcodec for Audio decode. We don't have
    # torchcodec in the image (and don't want it — extra ~200 MB). Disable
    # the auto-decoder and decode manually with soundfile below.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Gather candidates with basic quality filtering. Stream stops at
    # max_scan so we don't burn the whole split on a 70-clip sample.
    candidates: list[dict] = []
    for i, row in enumerate(ds):
        if i >= args.max_scan:
            break
        audio = row.get("audio") or {}
        # With decode=False the Audio feature yields {"bytes": b"...", "path": "..."}.
        # Fall back to the old decoded shape just in case.
        if "bytes" in audio and audio["bytes"] is not None:
            try:
                arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            except Exception:
                continue
            # sf.read returns (samples,) for mono, (samples, channels) for multi
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
        elif audio.get("array") is not None:
            arr = audio["array"]
            sr = audio["sampling_rate"]
        else:
            continue
        if sr is None or arr is None:
            continue
        duration = len(arr) / sr
        if duration < 1.0 or duration > 10.0:
            # Too short: not enough signal. Too long: disproportionate
            # decode cost for a smoke set.
            continue
        # FLEURS uses `transcription`, CV used `sentence`, Golos uses `transcription`.
        text = (row.get("transcription") or row.get("sentence") or row.get("text") or "").strip()
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

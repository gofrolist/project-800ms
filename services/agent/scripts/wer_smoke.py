"""One-shot WER smoke harness for the STT A/B experiment.

Reads a curated Russian eval set, feeds each utterance through both
STT services offline (no LiveKit, no Pipecat session lifecycle), computes
per-stack + per-category WER, and writes a result CSV. The committed
result CSV + this script together are the durable evidence for the
Phase A decision.

Usage (from inside the agent container on an L4 host):
    python scripts/wer_smoke.py

Environment overrides:
    EVAL_SET_DIR — default services/agent/eval/russian_eval_set
    RESULTS_CSV  — default services/agent/eval/results/<date>-wer-smoke.csv

Eval-set layout:
    services/agent/eval/russian_eval_set/
    ├── truth.jsonl          # one JSON per line: {audio: "clip-01.wav", text: "…", category: "clean"}
    ├── clip-01.wav          # 16 kHz mono WAV
    └── …

Categories (for per-category WER breakdown):
    clean            — studio-quality speech, game-adjacent vocab
    noisy            — background music / room ambiance
    game_specific    — character names, items, quest keywords
    stress           — stress-mark sensitive words (ruc vs ruč)
    short            — < 2s utterances (tests the min-duration filter)

Decision gate criteria (from plan Unit 3):
    Δ WER ≥ 5 points in one direction  → decisive, skip Phase B.
    Δ WER < 5 points or mixed per-cat  → inconclusive, proceed to Phase B.

See: docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md (Unit 3)
     docs/experiments/2026-04-19-stt-ab/phase-a-results.md (populated after run)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Eval set categories — see module docstring.
CATEGORIES = ("clean", "noisy", "game_specific", "stress", "short")

# Punctuation to strip before WER — Whisper emits ". , ? ! : ;" etc.,
# ground truth is punctuation-free. Counting these as errors would
# double-penalize Whisper for a stylistic difference, not a recognition
# mistake.
_PUNCT_CHARS = '.,?!:;"«»„“()[]{}—–-_/'

# A conservative digit→spelled-Russian map for 0-20 + tens. Whisper tends
# to emit digits for numbers in commands ("14"), truth has Russian words
# ("четырнадцать"). This won't cover prices / years / dates, which still
# count as real errors — we note the residual in phase-a-results.md.
_DIGITS_RU = {
    "0": "ноль",
    "1": "один",
    "2": "два",
    "3": "три",
    "4": "четыре",
    "5": "пять",
    "6": "шесть",
    "7": "семь",
    "8": "восемь",
    "9": "девять",
    "10": "десять",
    "11": "одиннадцать",
    "12": "двенадцать",
    "13": "тринадцать",
    "14": "четырнадцать",
    "15": "пятнадцать",
    "16": "шестнадцать",
    "17": "семнадцать",
    "18": "восемнадцать",
    "19": "девятнадцать",
    "20": "двадцать",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, expand small digits, collapse whitespace.

    Removes the stylistic noise that makes Whisper look much worse than
    it is on Golos (capitalization, punctuation, digit style). Leaves
    real recognition errors (wrong words, hallucinations, Latin
    transliteration of Cyrillic abbreviations) intact so they still
    show up as WER.
    """
    text = text.lower()
    # Strip punctuation by replacing with space (so "14К." → "14К " not "14К")
    for ch in _PUNCT_CHARS:
        text = text.replace(ch, " ")
    # Expand 0-20 digits to Russian words where they appear as standalone
    # tokens. Don't touch multi-digit numbers (years, prices) — covering
    # those correctly needs num2words and would hide real errors anyway.
    tokens = [_DIGITS_RU.get(t, t) for t in text.split()]
    return " ".join(tokens).strip()


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via simple Levenshtein on tokens.

    Avoids a heavy `jiwer` dep for this one-shot script. Reference:
    https://en.wikipedia.org/wiki/Word_error_rate
    """
    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Standard DP Levenshtein over word tokens.
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
    return d[len(ref)][len(hyp)] / len(ref)


def _load_wav_as_int16_pcm(path: Path) -> bytes:
    """Return int16 mono 16 kHz PCM bytes for the given WAV."""
    import soundfile as sf  # noqa: PLC0415 — heavy dep only needed here

    data, sr = sf.read(str(path), dtype="int16")
    if sr != 16_000:
        raise RuntimeError(f"{path} sample rate is {sr}, expected 16000")
    if data.ndim > 1:
        # Downmix to mono by averaging.
        data = np.mean(data, axis=1).astype(np.int16)
    return data.tobytes()


def _run_whisper(audio_bytes: bytes, audio_path: Path) -> tuple[str, float]:
    """Transcribe one utterance with faster-whisper. Returns (text, seconds)."""
    from faster_whisper import WhisperModel  # noqa: PLC0415

    global _whisper_singleton  # noqa: PLW0603
    if "_whisper_singleton" not in globals():
        _whisper_singleton = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

    audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    t = time.monotonic()
    segments, _ = _whisper_singleton.transcribe(audio_float, language="ru")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, time.monotonic() - t


def _run_gigaam(audio_bytes: bytes, audio_path: Path) -> tuple[str, float]:
    """Transcribe one utterance with GigaAM-v3. Returns (text, seconds).

    GigaAM's transcribe() expects a file path — it shells out to ffmpeg
    to decode. Pass the WAV path directly rather than the decoded bytes.
    """
    import gigaam  # noqa: PLC0415

    global _gigaam_singleton  # noqa: PLW0603
    if "_gigaam_singleton" not in globals():
        _gigaam_singleton = gigaam.load_model("v3_ctc")

    t = time.monotonic()
    result = _gigaam_singleton.transcribe(str(audio_path))
    text = result.text.strip() if hasattr(result, "text") else str(result).strip()
    return text, time.monotonic() - t


def _load_eval_set(eval_dir: Path) -> list[dict]:
    truth_path = eval_dir / "truth.jsonl"
    if not truth_path.exists():
        raise FileNotFoundError(
            f"{truth_path} not found. See docs/plans/2026-04-19-001-*.md for the eval-set format."
        )
    entries = []
    with truth_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(json.loads(line))
    # Validate shape
    for i, e in enumerate(entries):
        missing = {"audio", "text", "category"} - e.keys()
        if missing:
            raise ValueError(f"truth.jsonl entry {i} missing keys: {missing}")
        if e["category"] not in CATEGORIES:
            raise ValueError(
                f"truth.jsonl entry {i} has unknown category {e['category']!r}; "
                f"expected one of {CATEGORIES}"
            )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WER smoke for STT A/B.")
    parser.add_argument(
        "--eval-dir",
        default=os.environ.get(
            "EVAL_SET_DIR",
            str(Path(__file__).parent.parent / "eval" / "russian_eval_set"),
        ),
    )
    parser.add_argument(
        "--out",
        default=os.environ.get(
            "RESULTS_CSV",
            str(
                Path(__file__).parent.parent
                / "eval"
                / "results"
                / f"{datetime.date.today().isoformat()}-wer-smoke.csv"
            ),
        ),
    )
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Eval set: {eval_dir}")
    print(f"Results:  {out_path}\n")

    entries = _load_eval_set(eval_dir)
    print(
        f"Loaded {len(entries)} utterances across {len(set(e['category'] for e in entries))} categories.\n"
    )

    rows: list[dict] = []
    per_stack_totals: dict[str, list[float]] = defaultdict(list)
    per_category: dict[tuple[str, str], list[float]] = defaultdict(list)

    for i, entry in enumerate(entries, 1):
        audio_path = eval_dir / entry["audio"]
        print(f"[{i}/{len(entries)}] {entry['audio']} ({entry['category']})")
        try:
            audio_bytes = _load_wav_as_int16_pcm(audio_path)
        except Exception as exc:
            print(f"  skipped (audio load failed): {exc}")
            continue

        whisper_text, whisper_s = _run_whisper(audio_bytes, audio_path)
        gigaam_text, gigaam_s = _run_gigaam(audio_bytes, audio_path)

        whisper_wer = _wer(entry["text"], whisper_text)
        gigaam_wer = _wer(entry["text"], gigaam_text)

        per_stack_totals["whisper"].append(whisper_wer)
        per_stack_totals["gigaam"].append(gigaam_wer)
        per_category[("whisper", entry["category"])].append(whisper_wer)
        per_category[("gigaam", entry["category"])].append(gigaam_wer)

        rows.append(
            {
                "audio": entry["audio"],
                "category": entry["category"],
                "truth": entry["text"],
                "whisper_text": whisper_text,
                "whisper_wer": f"{whisper_wer:.4f}",
                "whisper_ms": f"{whisper_s * 1000:.1f}",
                "gigaam_text": gigaam_text,
                "gigaam_wer": f"{gigaam_wer:.4f}",
                "gigaam_ms": f"{gigaam_s * 1000:.1f}",
            }
        )
        print(f"  whisper: WER={whisper_wer:.3f}  ({whisper_s * 1000:.0f}ms)")
        print(f"  gigaam:  WER={gigaam_wer:.3f}  ({gigaam_s * 1000:.0f}ms)")

    # ── Write per-utterance CSV ─────────────────────────────────────
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stack, wers in per_stack_totals.items():
        print(f"  {stack:10s} mean WER: {np.mean(wers):.4f}  (n={len(wers)})")

    print("\nPer-category:")
    print(f"  {'category':<15s} {'whisper':>10s} {'gigaam':>10s} {'delta':>10s}")
    for cat in CATEGORIES:
        w = per_category.get(("whisper", cat), [])
        g = per_category.get(("gigaam", cat), [])
        if not w or not g:
            continue
        w_mean = float(np.mean(w))
        g_mean = float(np.mean(g))
        delta = g_mean - w_mean  # negative = gigaam is better
        print(f"  {cat:<15s} {w_mean:>10.4f} {g_mean:>10.4f} {delta:>+10.4f}")

    whisper_mean = float(np.mean(per_stack_totals["whisper"]))
    gigaam_mean = float(np.mean(per_stack_totals["gigaam"]))
    delta_pts = (whisper_mean - gigaam_mean) * 100  # positive = gigaam wins

    print(f"\nOverall Δ (whisper − gigaam): {delta_pts:+.2f} WER points")
    if delta_pts >= 5:
        print(
            "DECISION: Decisive GigaAM win → skip Phase B, adopt as default, keep Whisper 1 release."
        )
    elif delta_pts <= -5:
        print("DECISION: Decisive Whisper win → stop experiment, delete GigaAM work.")
    else:
        print("DECISION: Inconclusive → proceed to Phase B (blinded subjective A/B).")

    print(f"\nPer-utterance CSV: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

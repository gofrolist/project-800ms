"""Day-1 VRAM verification for the STT A/B experiment (Unit 1).

Loads the full runtime GPU footprint — Whisper-large-v3 (int8_float16) +
GigaAM-v3 + Silero VAD — and reports peak VRAM via `nvidia-smi`. The
agent runtime also sits alongside vLLM (Qwen-7B-AWQ) on the same L4 in
the current demo deployment; this script includes a note on how to
run it with vLLM resident too, but does not attempt to start vLLM here
(that's a separate process owned by the compose stack).

Ceiling: 22 GB on a 24 GB L4. If exceeded, pause the experiment —
either drop to distil-whisper or defer until a TRT/ONNX quantization
path for GigaAM is available.

Usage (from inside the agent container on an L4 host):
    python scripts/verify_vram.py

With vLLM already running on the same GPU:
    1. Start the full stack: `docker compose up -d`
    2. docker exec into the agent container
    3. Run this script

Output: prints a table + pass/fail verdict. Exits non-zero on ceiling
breach so it can be wired into CI / deploy gates.

See: docs/plans/2026-04-19-001-feat-stt-ab-experiment-plan.md (Unit 1).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# 24 GB L4, with ~2 GB margin for OS + CUDA context drift.
CEILING_MIB = 22 * 1024

WHISPER_MODEL = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
GIGAAM_MODEL = os.environ.get("GIGAAM_MODEL", "v3_ctc")


def _nvidia_smi_used_mib() -> int:
    """Return currently-used MiB on GPU 0."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    ).strip()
    return int(out.splitlines()[0])


def _row(label: str, mib: int, delta: int | None = None) -> None:
    delta_str = f" (+{delta} MiB)" if delta is not None else ""
    print(f"  {label:30s} {mib:>6} MiB{delta_str}")


def main() -> int:
    print("VRAM verification for STT A/B experiment")
    print(f"  Ceiling: {CEILING_MIB} MiB ({CEILING_MIB / 1024:.1f} GB)")
    print()

    # Baseline — whatever's already on the GPU (vLLM if running).
    baseline = _nvidia_smi_used_mib()
    _row("baseline (before loads)", baseline)

    # ── Whisper load ──────────────────────────────────────────────────
    print("\nLoading Faster-Whisper (large-v3, int8_float16)...")
    from faster_whisper import WhisperModel  # noqa: PLC0415 — heavy import

    t = time.monotonic()
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16")
    whisper_load_s = time.monotonic() - t
    after_whisper = _nvidia_smi_used_mib()
    _row(f"after Whisper ({whisper_load_s:.1f}s)", after_whisper, after_whisper - baseline)

    # ── GigaAM load ──────────────────────────────────────────────────
    print("\nLoading GigaAM-v3...")
    import gigaam  # noqa: PLC0415 — heavy import

    t = time.monotonic()
    gigaam_model = gigaam.load_model(GIGAAM_MODEL)
    gigaam_load_s = time.monotonic() - t
    after_gigaam = _nvidia_smi_used_mib()
    _row(f"after GigaAM ({gigaam_load_s:.1f}s)", after_gigaam, after_gigaam - after_whisper)

    # ── Silero VAD ──────────────────────────────────────────────────
    # VAD runs on CPU in the Pipecat pipeline (see services/agent/pipeline.py),
    # so its footprint is negligible here. Still load it to mirror the
    # runtime configuration — the onnx runtime has its own CPU/GPU context.
    print("\nLoading Silero VAD analyzer (CPU path)...")
    from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: PLC0415

    t = time.monotonic()
    _vad = SileroVADAnalyzer()  # noqa: F841 — held only to keep in-process
    vad_load_s = time.monotonic() - t
    after_vad = _nvidia_smi_used_mib()
    _row(f"after VAD ({vad_load_s:.1f}s)", after_vad, after_vad - after_gigaam)

    total_delta = after_vad - baseline
    print(f"\n  TOTAL STT+VAD footprint: {total_delta} MiB (~{total_delta / 1024:.2f} GB)")
    print(f"  PEAK resident: {after_vad} MiB (~{after_vad / 1024:.2f} GB)")

    if after_vad > CEILING_MIB:
        print(
            f"\nFAIL: peak {after_vad} MiB exceeds ceiling {CEILING_MIB} MiB.\n"
            "  Pause Unit 1. Consider: drop to distil-whisper, defer until\n"
            "  TRT/ONNX quant for GigaAM, or run Phase A offline only."
        )
        return 1

    print(f"\nPASS: peak {after_vad} MiB within ceiling {CEILING_MIB} MiB.")
    # Keep references alive until exit so Python doesn't GC mid-measure.
    _ = (whisper, gigaam_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

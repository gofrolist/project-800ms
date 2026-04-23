---
title: "coqui-tts needs the [codec] extra on PyTorch 2.9+ (torchcodec dependency split)"
date: 2026-04-22
module: services/agent
component: deps
category: build-errors
problem_type: build_error
tags: [coqui-tts, torchcodec, torchaudio, pytorch, xtts, pyproject, extras]
related_components: [services/agent/pyproject.toml, services/agent/uv.lock]
resolution_type: dep_extra
---

# `coqui-tts[codec]` needed for PyTorch 2.9+ audio I/O

## Symptom

After fixing the `transformers<5` pin (see related solutions), XTTS
preload got further before crashing:

```
XTTS pre-load failed — refusing to start agent
...
  File "/opt/venv/lib/python3.12/site-packages/TTS/__init__.py", line 46, in <module>
    raise ImportError(TORCHCODEC_IMPORT_ERROR)
ImportError:
From Pytorch 2.9, the torchcodec library is required for audio IO, but
it was not found in your environment. You can install it with Coqui's
`codec` extra:
```
pip install coqui-tts[codec]
```
```

Helpful error text — coqui-tts tells you exactly what to do. But the
pyproject.toml that works fine with `pip install coqui-tts` in a quick
test doesn't carry the extra, so the Docker build silently produces an
image that crashlooms at preload.

## Root cause

PyTorch 2.9 split audio I/O out of `torchaudio` into a new `torchcodec`
package. `torchaudio.load()` / `.save()` now **require** `torchcodec`
at runtime to actually decode files; without it, `torchaudio` imports
OK but raises when you try to load audio.

coqui-tts 0.27.5 doesn't depend on `torchcodec` in its base requirements
— it only advertises the `[codec]` extra. Downstream consumers who just
add `"coqui-tts>=0.27.5"` to their pyproject get an install that looks
clean but explodes on any `TTS.api.TTS.tts()` call that uses a
`speaker_wav` (which XTTS v2 always does for voice cloning).

Our `uv.lock` resolved `torch 2.11.0` via `gigaam[torch]`, well above
the 2.9 threshold. So we hit this reliably every deploy after the
transformers pin let the preload get past `from TTS.api import TTS`.

## Fix

Change the dep spec in `services/agent/pyproject.toml` from:

```toml
"coqui-tts>=0.27.5",
```

to:

```toml
"coqui-tts[codec]>=0.27.5",
```

The `[codec]` extra pulls `torchcodec==0.11.1` (4 MB wheel). No other
code changes needed.

Ref: commit `7784e33`.

Also verify the Ubuntu base image ships FFmpeg libraries torchcodec can
dlopen against: `ffmpeg libavutil.so.58` is sufficient on Ubuntu 24.04.
Additionally, `libpython3.12` is required for torchcodec's FFmpeg 6
loader — see [the libpython3.12 solution](./libpython-shared-lib-missing-ubuntu-apt.md).

## Why the lock resolved without the extra

`uv` / pip don't know which extras you need unless you list them. The
base `coqui-tts` install succeeds because the core library imports OK —
the runtime dependency is lazy, only evaluated when `torchaudio.load()`
is called. Static dep resolution has no visibility into that deferred
import.

## When this recurs

Any time PyTorch ships a major feature split:

- PyTorch 2.9 → `torchcodec` (audio/video I/O moved out of `torchaudio`)
- PyTorch 2.x → `torchtext` (already done in prior versions)
- Future PyTorch releases will likely continue splitting specialized
  functionality into side packages with `[codec]`/`[extra]`-style opt-ins.

Pattern signal: the library you depend on says `"From PyTorch X.N, foo
is required"` in its import error. That's the library telling you to
add an extra; the error text is machine-readable enough that a
Dockerfile smoke test (`python -c "import library"`) will surface it
pre-deploy. We added that smoke test in commit `28286ba`.

## What NOT to do

- **Don't install torchcodec separately** (`"torchcodec>=0.11"` as a
  top-level dep). The `[codec]` extra is coqui-tts's supported
  integration path and pins a compatible version. Adding it directly
  risks version drift from what coqui-tts expects.
- **Don't fall back to CPU-only decode paths** in torchaudio. Those
  exist but are slow and the decode errors are opaque; better to fix
  the dep.

## Related solutions

- [`transformers<5` override for coqui-tts](./transformers-v5-isin-mps-friendly-coqui-tts.md)
- [`libpython3.12` apt package for torchcodec FFmpeg loader](./libpython-shared-lib-missing-ubuntu-apt.md)

All three hotfixes for the XTTS rollout: `938cc44` (transformers),
`7784e33` (this one), `c4edb63` (libpython3.12).

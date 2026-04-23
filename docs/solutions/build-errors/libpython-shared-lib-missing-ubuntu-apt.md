---
title: "libpython3.12.so.1.0 missing from Ubuntu 24.04 apt (torchcodec FFmpeg 6 loader)"
date: 2026-04-22
module: services/agent
component: docker
category: build-errors
problem_type: build_error
tags: [docker, ubuntu-24.04, ffmpeg, torchcodec, libpython, shared-library, coqui-tts, xtts]
related_components:
  [
    services/agent/Dockerfile.base,
    services/agent/models.py,
    services/agent/xtts_tts.py,
  ]
resolution_type: dockerfile_fix
---

# `libpython3.12.so.1.0` missing from Ubuntu apt — torchcodec dlopen failure

## Symptom

Agent container crashloops on first XTTS synth with:

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment. We support
     versions 4, 5, 6, 7, and 8 ...
  2. The PyTorch version (2.11.0+cu130) is not compatible ...
  3. Another runtime dependency; see exceptions below.

[start of libtorchcodec loading traceback]
FFmpeg version 6:
OSError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory
```

XTTS preload (`models.load_xtts`) completes without issue — the symptom only
surfaces during the first actual synth call, when `torchaudio.load()` lazy-
imports `torchcodec` to decode the voice-clone reference WAV.

## Why the error is misleading

The torchcodec error message walks through every FFmpeg version it has a
loader for (4-8) and surfaces the dlopen error for each. FFmpeg 8 / 7 / 5
/ 4 fail with `libavutil.so.N: cannot open shared object file` because
those versions aren't installed on Ubuntu 24.04 — those failures are
**expected** and not the problem.

The real clue is buried in the **FFmpeg 6** attempt, which is the version
Ubuntu 24.04 ships: that loader fails with `libpython3.12.so.1.0: cannot
open shared object file`. Python's own shared library is not present.

First instinct on seeing this: "Docker is misconfigured" or "FFmpeg is
broken." Neither is true. The actual fix is one apt package.

## Root cause

Ubuntu 24.04's `python3.12` apt package ships a statically-linked Python
interpreter **without** `libpython3.12.so.1.0`. Most Python workloads
don't need the shared lib — it's only required when a native extension
wants to `dlopen(libpython3.12.so.1.0)` at runtime to call back into
Python's C API.

`torchcodec`'s FFmpeg 6 loader does exactly this. Install chain:

```
TTS.api.tts()
  → torchaudio.load(ref.wav)
    → torchaudio._torchcodec.load_with_torchcodec()
      → from torchcodec.decoders import AudioDecoder
        → torchcodec._core.ops.load_torchcodec_shared_libraries()
          → ctypes.CDLL(libtorchcodec_core6.so)
            → dlopen() needs libpython3.12.so.1.0  ← missing
```

## Fix

Install `libpython3.12` alongside `python3.12` in the runtime stage of
the agent base image:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 libpython3.12 \
        ffmpeg libsndfile1 espeak-ng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
```

`libpython3.12` is `<2 MB` — adding it to the runtime apt install has
negligible image-size impact.

Ref: `services/agent/Dockerfile.base` commit `c4edb63`.

## When this recurs

Any Ubuntu-based image that runs a native extension dlopen-ing
`libpython3.N.so.1.0`. The pattern is:

- coqui-tts + PyTorch 2.9+ → torchaudio → torchcodec → FFmpeg-N loader
- (likely) other PyTorch audio/video libraries migrating to torchcodec
- (possibly) any native extension that wraps a Python-callback library

**When you see `libpython3.N.so.1.0: cannot open shared object file`
in a torchcodec or FFmpeg-related traceback** — the fix is to add
`libpython3.N` to the Dockerfile apt install.

## What NOT to do

- **Don't install a different FFmpeg version.** Ubuntu 24.04's FFmpeg 6
  is fine; the problem is Python's shared lib, not FFmpeg.
- **Don't switch to a slim Python base image** (`python:3.12-slim`) —
  those also statically link Python by default; the issue recurs.
- **Don't vendor `libpython3.12.so.1.0` by copying it from a dev image.**
  The apt package is the supported path; copying shared libs across
  images breaks other native extensions that rely on apt's dependency
  metadata.

## Related solutions

- [coqui-tts + torchcodec `[codec]` extra](./torchcodec-codec-extra-pytorch-29.md)
- [`transformers<5` override for coqui-tts](./transformers-v5-isin-mps-friendly-coqui-tts.md)

Both were discovered in the same XTTS v2 rollout (commits 938cc44,
7784e33, c4edb63 on main).

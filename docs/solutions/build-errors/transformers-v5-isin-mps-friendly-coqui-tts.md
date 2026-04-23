---
title: "transformers 5.x removed isin_mps_friendly — coqui-tts ImportError at agent boot"
date: 2026-04-22
module: services/agent
component: deps
category: build-errors
problem_type: build_error
tags: [transformers, coqui-tts, xtts, pydantic, uv, dep-override, major-version-drift]
related_components: [services/agent/pyproject.toml, services/agent/uv.lock]
resolution_type: uv_override
---

# `transformers 5.x` removed `isin_mps_friendly` — coqui-tts ImportError

## Symptom

Agent container crashlooped on XTTS preload with:

```
XTTS pre-load failed — refusing to start agent
Traceback (most recent call last):
  File "/app/models.py", line 188, in load_xtts
    from TTS.api import TTS
  File "/opt/venv/lib/python3.12/site-packages/TTS/__init__.py", line 35, in <module>
    from TTS.tts.configs.xtts_config import XttsConfig
  File ".../TTS/tts/models/xtts.py", line 15, in <module>
    from TTS.tts.layers.xtts.gpt import GPT
  File ".../TTS/tts/layers/xtts/gpt.py", line 10, in <module>
    from TTS.tts.layers.tortoise.autoregressive import (...)
  File ".../TTS/tts/layers/tortoise/autoregressive.py", line 12, in <module>
    from transformers.pytorch_utils import isin_mps_friendly as isin
ImportError: cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'
```

Happens at the first `from TTS.api import TTS` — not at synth time.
main.py's XTTS preload block catches the exception and `sys.exit(3)`,
which Docker's restart policy turns into a crashloop.

## Root cause

`coqui-tts 0.27.5` (January 2026) imports `isin_mps_friendly` from
`transformers.pytorch_utils` via the Tortoise autoregressive decoder
(inherited from the deprecated upstream Coqui TTS code). The symbol was
**removed** in `transformers 5.0`.

The agent's `uv.lock` had pipecat's transitive deps resolve
`transformers==5.5.0`. Pipecat itself doesn't require `transformers>=5`
— the version drift was opportunistic (no upper bound in any direct
dep). Adding `coqui-tts` didn't automatically correct the drift because
coqui-tts 0.27.5 also has no explicit upper bound on transformers (its
dep spec was a plain `transformers`).

## Fix

Pin `transformers<5.0` via `[tool.uv] override-dependencies` in
`services/agent/pyproject.toml`:

```toml
[tool.uv]
override-dependencies = [
    "onnx>=1.21.0",                  # existing
    "transformers>=4.46,<5.0",       # NEW — coqui-tts compat
]
```

`uv lock` resolves to `transformers 4.57.6` under this constraint.
Verified by running the full agent test suite (160 tests) — none regress
on the 4.x line. Pipecat 0.0.108 + gigaam both accept transformers 4.x
(neither pins a >=5 floor).

Ref: commit `938cc44`.

## Why this class of failure is sneaky

The breakage surfaces as an ImportError **on the container's startup
path**, not during test or local dev. The agent test suite mocks out
`TTS.api.TTS` (see `test_xtts_tts.py::TestLoadXtts._patch_tts_import`)
because coqui-tts isn't installed on CPU-only CI. The lockfile resolved
cleanly. Lint passed. CI passed. The import chain only blew up when the
real `TTS.api` module executed inside the built Docker image.

Post-fix CI smoke test (commit `28286ba`): the agent Docker build job
now runs `python -c "from TTS.api import TTS"` after building the
image. This catches the same class of failure at PR time — all three
XTTS rollout hotfixes (this one, `[codec]`, `libpython3.12`) were
import-time failures a one-line smoke would have caught pre-merge.

## When this recurs

Any transitive dep bump across a major version boundary where a library
downstream imports a non-public or deprecated symbol. Specifically:

- **Transformers minor / major bumps** (they remove internal utilities
  often) + libraries using the Tortoise TTS family (`coqui-tts`,
  `tortoise-tts`, some TTS research forks).
- **`huggingface_hub` minor bumps** — similar pattern, several utilities
  have been relocated without deprecation.

Pattern signal: `ImportError: cannot import name 'X' from 'Y'` where
`X` is a `_` or `utils`-style name. Grep the upstream library's
changelog for the removed symbol before overriding.

## What NOT to do

- **Don't drop the override comment.** `pyproject.toml` now carries a
  paragraph explaining why `transformers<5.0` is pinned. A future
  engineer cleaning up "stale overrides" needs that context to avoid
  regressing (would cause the exact same crashloop).
- **Don't bump the upper bound speculatively.** coqui-tts 0.27.x is
  the last affected series; watch their changelog for the upstream
  Tortoise cleanup. When it lands, remove the override together with
  the coqui-tts version bump in one commit, not separately.
- **Don't pin specific minor.** `<5.0` lets uv track security patches
  within the 4.x line.

## Related solutions

- [`coqui-tts[codec]` extra for PyTorch 2.9+ torchaudio](./torchcodec-codec-extra-pytorch-29.md)
- [`libpython3.12` apt package for torchcodec FFmpeg loader](./libpython-shared-lib-missing-ubuntu-apt.md)

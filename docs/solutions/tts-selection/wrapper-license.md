---
title: "Qwen3-TTS community wrapper — license record"
date: 2026-04-21
module: infra/qwen3-tts-wrapper
component: tts
category: tts-selection
problem_type: best_practice
tags: [tts, qwen3, licensing, vendoring]
resolution_type: license_record
---

# Qwen3-TTS community wrapper — license record

## Vendored dependency

- **Repo:** [dingausmwald/Qwen3-TTS-Openai-Fastapi](https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi)
- **Pinned commit:** `eb14f6e6a50445cf442979abb9203ff0d5042c43`
- **Vendored on:** 2026-04-21
- **Local location:** `infra/qwen3-tts-wrapper/`

## License determination

- **License:** Apache License, Version 2.0
- **Source of truth:** `infra/qwen3-tts-wrapper/LICENSE` (verbatim copy from upstream `LICENSE` at the pinned commit — verified SPDX-License-Identifier headers in `api/main.py` and other files confirm `Apache-2.0`).
- **Commercial use:** permitted. Redistribution: permitted. Modification: permitted.
- **Compatibility with project repo policy:** project follows MIT/Apache-only for self-hosted components (see Plan scope). Apache 2.0 satisfies this.

## Attribution requirements

Apache 2.0 requires:
1. Retain the copyright notice and the `LICENSE` file when redistributing. **Satisfied** — `infra/qwen3-tts-wrapper/LICENSE` is a verbatim copy, not stripped.
2. State changes if the work is modified. **No modifications as of 2026-04-21.** Any future local patches are tracked in `infra/qwen3-tts-wrapper/README.md` under a `## Local patches` section (created when needed).
3. Include attribution to "Alibaba Qwen Team" (the `pyproject.toml` `authors` field credits them as the original Qwen3-TTS creators; the wrapper author is documented in the upstream README but not separately credited in the source files themselves).

## Underlying model weights license

Qwen3-TTS model weights (separate from this wrapper code) are hosted on Hugging Face (`Qwen/Qwen3-TTS-12Hz-0.6B-Base` and `Qwen/Qwen3-TTS-12Hz-1.7B-Base`). Their license is set by Alibaba on the HF model card — **confirm before the sidecar first-downloads weights during Unit 4 deploy**. Apache 2.0 on the inference wrapper does not retroactively license the model weights.

If the Qwen3 model weights turn out to be non-commercial or restrictively licensed, the entire Qwen3 track is dropped (R23 abort gate) and the experiment reduces to Piper-vs-Silero. The wrapper license alone does not resolve this — keep it as a Unit 4 pre-deploy check item.

## Upstream dependencies (license check — indirect)

The wrapper pulls pip dependencies (see `infra/qwen3-tts-wrapper/requirements.txt`). All top-level deps are permissive (MIT/BSD/Apache) — fastapi, uvicorn, pydantic, torch, transformers, librosa, soundfile, pydub, num2words. No known GPL or AGPL transitive dep at the pinned versions. Flash-attn (installed separately in the Dockerfile) is BSD-3-Clause. No license concerns.

## Revalidation cadence

- **On upstream commit bump:** re-read the upstream LICENSE file at the new SHA, update this record's `Pinned commit` and `Vendored on` fields, re-verify no relicensing has occurred.
- **On Qwen3 model weight update:** re-check the HF model card license.
- **On major dependency bump:** re-check transitive license tree via `pip-licenses` or equivalent if concerns arise.

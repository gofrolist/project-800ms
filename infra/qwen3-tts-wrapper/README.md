# Qwen3-TTS OpenAI-compatible sidecar (vendored)

Vendored community wrapper that exposes Qwen3-TTS via an OpenAI-compatible HTTP API, so Pipecat's upstream `OpenAITTSService` can talk to it without custom adapter code on the agent side.

## Provenance

- **Upstream repo:** [dingausmwald/Qwen3-TTS-Openai-Fastapi](https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi)
- **Pinned commit:** `eb14f6e6a50445cf442979abb9203ff0d5042c43` (2026-04-21 snapshot)
- **License:** Apache License 2.0 (see `LICENSE` in this directory — verbatim copy from upstream). Compatible with our MIT/Apache-only repo policy.
- **Vendored on:** 2026-04-21, as part of Phase 0 R23 of `docs/plans/2026-04-21-001-feat-tts-abstraction-piper-silero-qwen3-plan.md`.

## Why vendored, not submoduled

- A `git submodule` would make the commit SHA changeable by downstream git operations and makes "edit-to-patch" flow clumsy. Vendoring keeps a read/write copy in-tree — the single developer + Claude Code can patch the wrapper directly if upstream is abandoned or regresses against our usage. The plan (R8a) explicitly names this ownership requirement.
- On bump: clone the upstream repo at a newer SHA, diff against our vendored copy, reconcile any local patches, bump the commit SHA recorded above.

## What was vendored

Selective copy — not the full 17 MB upstream repo. **Kept:**

```
api/             # FastAPI server with OpenAI-compat router (~224 KB)
qwen_tts/        # Model inference package (~464 KB)
config.yaml      # Model/voice/optimization config (see "Config" below — needs editing for our deploy)
Dockerfile       # NVIDIA CUDA variant (pytorch:2.6.0-cuda12.6)
LICENSE          # Apache 2.0 verbatim
MANIFEST.in      # Python packaging manifest
pyproject.toml   # Python package metadata
requirements.txt # Pinned pip deps
extended_warmup.py  # Optional warmup helper referenced by Dockerfile
```

**Skipped intentionally:**
- `.git/` (16 MB of upstream history — we re-record provenance in this README)
- `examples/`, `finetuning/`, `install/`, `docs/`, `patches/` (upstream dev helpers irrelevant to our runtime)
- `gradio_voice_studio.py` (38 KB gradio UI, unused)
- `bench_*.py`, `bench_*.txt`, `benchmark_*.py`, `test_*.py`, `verify_*.py`, `optimization_verification.txt` (benchmarking & upstream test harness)
- `Dockerfile.rocm`, `Dockerfile.vllm`, `docker-compose.rocm.yml`, `docker-compose.yml` (alternate backends not used)
- `.github/`, `.env.example`, `.gitignore`, `start_server.sh`, misc `*.md` upstream docs

If a skipped path becomes relevant (e.g., we want to use the vLLM backend), re-vendor that piece and update the kept/skipped list above.

## Request/response contract (verified against source at pinned SHA)

**Endpoint:** `POST /v1/audio/speech`
**Router:** `api/routers/openai_compatible.py:225`
**Request schema:** `api/structures/schemas.py:44` (`OpenAISpeechRequest`)

| Field | Type | Our usage |
|---|---|---|
| `model` | `str` | Send `"tts-1-ru"` to trigger Russian output. Accepted aliases are built from `LANGUAGE_CODE_MAPPING` at `api/routers/openai_compatible.py:95-113`: `"tts-1"`, `"tts-1-hd"`, `"qwen3-tts"`, and `"tts-1-<lang>"` / `"tts-1-hd-<lang>"` for `lang ∈ {en, zh, ja, ko, de, fr, es, ru, pt, it}`. Our default: `tts-1-ru`. |
| `input` | `str` | The text to synthesize. |
| `voice` | `str` | One of Pipecat's built-in OpenAI voice names (`alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`) — the wrapper maps these at `openai_compatible.py:163-170` to its internal Qwen3 voices (`Vivian`, `Ryan`, `Serena`, `Aiden`, `Eric`, `Dylan`). Also accepts raw Qwen3 voice names (e.g., `Ryan`) and cloned-voice form (`clone:MyVoice`). **This mapping is the resolution for the P1 `OpenAITTSService.VALID_VOICES` whitelist finding from the ce-doc-review — Pipecat's enforced voice whitelist passes the wrapper's validator.** |
| `response_format` | `Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]` | **`pcm` is supported** (see `api/services/audio_encoding.py:146-147`) — matches Pipecat's `OpenAITTSService` hardcoded `response_format='pcm'`. |
| `speed` | `float`, default 1.0 | Unused; keep default. |
| `stream` | `bool`, default false | **Set to `true`** for lowest TTFB — yields PCM chunks as the model generates. When `stream=true` AND `response_format=wav`, wrapper silently rewrites `wav → pcm` (line 302-304). Server returns `StreamingResponse` in both streaming and non-streaming modes. |
| `language` | `Optional[str]` | Optional — if the model alias already encodes language (e.g. `tts-1-ru`), this is ignored. |

**Response:** streaming or buffered `audio/pcm` bytes at the model's native sample rate (typically 24 kHz for Qwen3-TTS 0.6B — verify in R23 probe). Pipecat's helper `_stream_audio_frames_from_iterator` resamples to `self.sample_rate` if they differ.

**Voice mapping table** (from `openai_compatible.py:163-170`, verbatim):

| OpenAI name | Qwen3 voice | Language hint (from upstream `config.yaml` `voices:` list) |
|---|---|---|
| `alloy` | `Vivian` | Chinese |
| `echo` | `Ryan` | English |
| `fable` | `Serena` | Chinese |
| `nova` | `Aiden` | Japanese |
| `onyx` | `Eric` | Chinese |
| `shimmer` | `Dylan` | Chinese |

The "language hint" is the voice's primary trained language. Qwen3-TTS is a multilingual model — setting `model=tts-1-ru` overrides the voice's trained language and renders Russian. Cross-lingual quality varies per voice; recommend `echo` (Ryan) for Russian as a starting default since it's the only English-trained voice in the list and English training is closer to Russian phonology than Chinese/Japanese. Verify empirically during R23 smoke.

## Endpoints (complete list)

From `api/routers/openai_compatible.py`:

- `POST /v1/audio/speech` — primary TTS endpoint (above).
- `GET /v1/models` — list supported model aliases. Pipecat doesn't call this; useful for debugging.
- `GET /v1/voices` — list built-in + cloned voices. Debugging only.
- `POST /v1/voices/clone` — voice cloning (out of scope — our plan Non-Goals explicitly excludes voice cloning).
- `GET /health` — healthcheck used by the compose healthcheck block.

## Deps, pinning, CUDA assumptions

**Python version:** `>=3.9` per `pyproject.toml`. The container uses whatever Python comes with `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` (Python 3.11 at the time of the pinned SHA).

**Requirements (`requirements.txt`):**

| Dep | Pin | Why it matters |
|---|---|---|
| fastapi | `>=0.104.0` | HTTP server |
| uvicorn[standard] | `>=0.24.0` | ASGI server |
| pydantic | `>=2.0` | Request schema validation |
| soundfile, librosa | `>=0.12.0` / `>=0.10.0` | Audio I/O |
| pydub, ffmpeg-python | `>=0.25.0` / `>=0.2.0` | Format conversion (mp3/opus/aac) — we only use pcm so these could be dropped in a slim variant |
| torch | `>=2.4.0` | Model inference |
| transformers | `>=4.46.0` | Model loader |
| accelerate | `>=0.25.0` | Device placement |
| pyyaml | `>=6.0` | Config loader |
| num2words | `>=0.5.0` | Text normalization |

**Also pip-installed inside Dockerfile:** `flash-attn --no-build-isolation` (NVIDIA CUDA flash-attention kernels — fat wheel, can fail on unusual CUDA versions). If this install step becomes flaky, patch the Dockerfile to make it optional or skip entirely — the wrapper supports a non-flash backend (see `api/backends/pytorch_backend.py`) selectable via `TTS_BACKEND=pytorch` env.

**CUDA assumptions:** base image is `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`. Our agent image uses CUDA 13.2.1 — **the sidecar does not share the agent's image**, so CUDA version divergence is intentional and fine. The L4 GPU is backward-compatible (driver-level CUDA 13 can run CUDA 12 images). Keep the sidecar's base image pinned independently — do not try to unify with the agent's Dockerfile.base.

**Build-time network deps:** pip pulls from PyPI + PyTorch index; flash-attn wheel downloads from pypi + GitHub releases. Pre-cache for offline builds if egress becomes an issue.

**Runtime model download:** On first boot the wrapper fetches Qwen3-TTS weights from Hugging Face (`Qwen/Qwen3-TTS-12Hz-0.6B-Base` by default — see `config.yaml`). Weights cache to `HF_HOME` (default `/root/.cache/huggingface`). The compose service mounts `hf_cache_qwen3:/root/.cache/huggingface` to persist across restarts.

## Config (critical — must be overridden before first boot)

**`config.yaml` upstream has hardcoded local paths:**

```yaml
models:
  0.6B-Base:
    hf_id: /home/pawel/qwen3-tts/models/0.6B-Base   # upstream maintainer's local path — must override
  1.7B-Base:
    hf_id: Qwen/Qwen3-TTS-12Hz-1.7B-Base            # this one is an HF id, OK
```

Our plan deploys the 0.6B variant (R8 committed). We must either:

1. **Edit the vendored `config.yaml`** to point `0.6B-Base.hf_id` at an HF id (e.g., `Qwen/Qwen3-TTS-12Hz-0.6B-Base`). Check HF Hub for the actual id name at Unit 4 implementation time.
2. **Or** mount a custom `config.yaml` into the container at `/root/qwen3-tts/config.yaml` (the path the container looks up per Dockerfile). Cleaner for config-only changes.

**Our recommended approach:** option 1 (edit vendored copy) because the config is small and lives in version control, making the deploy self-contained.

## Lifecycle invariants relevant to our use

- **Model loaded lazily at first request** unless `TTS_WARMUP_ON_START=true`. Our compose healthcheck's `start_period: 300s` tolerates cold first synth; subsequent requests reuse the loaded model.
- **Single worker by default** (`WORKERS=1`). Do NOT scale up — the model is GPU-resident and multi-worker would multiply VRAM.
- **Keepalive** (`GPU_KEEPALIVE_INTERVAL`) sends periodic dummy synths to keep the CUDA context warm. Default off; enable if we observe cold-cache TTFB regressions.
- **CORS** accepts `*` by default. The sidecar is bound to `127.0.0.1` in compose, so no real exposure, but we can tighten `CORS_ORIGINS` if paranoid.
- **Logging** is stdout — compose `docker compose logs qwen3-tts` is the observability path (remember: no Prometheus metrics surface in scope per plan).

## Failure modes to know about (from source reading)

| Trigger | Observed behavior | Our mitigation |
|---|---|---|
| Unknown `model` name | HTTP 400 `invalid_model` with supported-list in detail | Agent always sends `tts-1-ru` which is in `MODEL_MAPPING` |
| Empty `input` after text normalization | HTTP 400 `invalid_input` | Upstream LLM never produces fully-empty output; acceptable error surface |
| `clone:` voice prefix without name | HTTP 400 `invalid_voice` | We don't use clone voices (out of scope) |
| Backend not ready | Blocks until backend init completes; no timeout | Compose `start_period: 300s` masks first-boot; subsequent requests are fast |
| Model OOM mid-request | Python exception propagates as HTTP 500 | Matches Pipecat's `OpenAITTSService` expectation (generic HTTP error → `ErrorFrame`) |
| flash-attn kernel mismatch (driver/toolkit skew) | Startup crash | Fall back to `TTS_BACKEND=pytorch` in the compose service env if this surfaces |

## How to patch if upstream is abandoned

1. Identify the file under `api/` or `qwen_tts/` that needs fixing (the source is in-tree — grep freely).
2. Make the fix directly in `infra/qwen3-tts-wrapper/` — no upstream PR needed.
3. Bump the "last-modified" note at the top of this README + record the reason for the local deviation in a `## Local patches` section below (create when needed).
4. Rebuild the compose service: `docker compose --profile tts-qwen3 build qwen3-tts && docker compose --profile tts-qwen3 up -d qwen3-tts`.

## Local patches

Patches applied on top of upstream commit `eb14f6e6a50445cf442979abb9203ff0d5042c43`:

1. **Bearer-token auth on `/v1/audio/speech`** (security/S1) — upstream accepts all requests without credentials; our compose internal network would expose the sidecar to every container. Added a simple dependency (`api/security.py::verify_api_key`) that validates the `Authorization: Bearer <token>` header against the `TTS_API_KEY` env var. Backwards-compatible: when `TTS_API_KEY` is unset the dependency is a no-op, so operators who haven't opted in keep upstream behavior.
2. **`/health` returns 503 when model not ready** (reliability/R2) — upstream returned 200 with `status=initializing` even while the backend was still loading weights; the compose healthcheck marked the container healthy before synth worked. Patched `api/main.py::health_check` to raise `HTTPException(status_code=503, ...)` when `backend.is_ready()` is False, so the healthcheck correctly stays in the "starting" state during first-boot weight download.
3. **CORS tightened to loopback by default** (security/S4) — upstream defaulted to `CORS_ORIGINS="*"` with `allow_credentials=True` on an unauthenticated endpoint; we default to `"http://localhost"` in `api/main.py`. Operators can restore the wildcard explicitly via env; the default stops being unsafe.
4. **Pydub-absent compressed-format fallback raises instead of silent** (adversarial/ADV-004) — upstream's compressed-audio path (mp3/opus/aac/flac) silently fell through to `convert_to_wav` when pydub was missing, returning WAV bytes behind an mp3/opus Content-Type header. Pipecat's downstream interpreted the RIFF header as raw PCM and rendered it as audio artifacts. Changed to raise `RuntimeError("pydub is required for <fmt> encoding")` so the failure is loud. `requirements.txt` pins pydub so the normal path is unaffected.
5. **VOICE_MAPPING canonicalised to `api/voice_mapping.py`** (adversarial/ADV-005) — upstream had two divergent copies of the OpenAI→Qwen voice table. The router's non-streaming path used the correct table; `api/backends/optimized_backend.py::generate_speech_streaming` hardcoded a different table referencing voices (`Sophia`, `Isabella`, `Lily`) that don't exist in the model catalog, so streaming requests for those OpenAI aliases silently fell back to defaults. Extracted the canonical table to `api/voice_mapping.py` and imported it from both call paths.

## References

- Plan R8a (wrapper pinning + vendoring rationale): `docs/plans/2026-04-21-001-feat-tts-abstraction-piper-silero-qwen3-plan.md`
- Plan R23 (structural readthrough requirement): same plan, Unit 1
- Upstream wrapper: https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi
- Qwen3-TTS model (Alibaba): https://github.com/QwenLM/Qwen3-TTS
- Pipecat `OpenAITTSService` (what the agent-side uses to talk to this sidecar): `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/openai/tts.py`

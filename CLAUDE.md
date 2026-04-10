# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time voice assistant: browser ↔ LiveKit (WebRTC) ↔ Pipecat agent (VAD → Whisper STT → LLM → TTS) with a FastAPI token-minting backend. Target: first audio out <800ms after end-of-speech.

## Architecture

```
Browser (React + LiveKit SDK)  ──WebRTC──►  LiveKit SFU (:7880)
                                                │
FastAPI (:8000)                          Pipecat Agent
  POST /sessions → mint LiveKit JWT       ├─ Silero VAD (CPU)
  GET  /health                            ├─ Faster-Whisper STT (GPU)
                                          ├─ LLM → vLLM (:8001) or external
                                          └─ Piper TTS (CPU)
```

Three deployable units: `apps/api` (FastAPI), `services/agent` (Pipecat pipeline), `apps/web` (React SPA). Infrastructure in `infra/` (Docker Compose + Terraform for AWS GPU spot).

## Commands

### API (apps/api) — Python 3.14, uv
```bash
cd apps/api
uv sync                     # install deps
uv run pytest -v            # run all tests
uv run pytest tests/test_main.py::test_create_session -v  # single test
ruff check .                # lint
ruff format .               # format
```

### Agent (services/agent) — Python 3.12, uv, CUDA required
```bash
cd services/agent
uv sync                     # install deps
uv run pytest -v            # tests (limited — CUDA deps at import)
ruff check .                # lint
ruff format .               # format
```

### Web (apps/web) — React 19, Vite, bun
```bash
cd apps/web
bun install                 # install deps
bun run dev                 # dev server :5173
bun run build               # tsc -b && vite build
```

### Full stack (Docker)
```bash
cp infra/.env.example infra/.env  # edit secrets first
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml logs -f vllm  # wait for model load
curl http://localhost:8000/health
```

### Pre-commit
```bash
pre-commit install
pre-commit run --all-files  # ruff, gitleaks, file hygiene, actionlint
```

## Key Design Decisions

- **Dynamic rooms**: each /sessions call creates a unique room and dispatches an agent pipeline via POST to the agent's HTTP server (:8001/dispatch). Rooms are isolated — callers can't hear each other.
- **Agent token minting**: agent mints a LiveKit JWT per room (30min TTL). API mints caller tokens (15min TTL).
- **Whisper hallucination filter**: `stt_filter.py` drops segments by statistical properties (no_speech_prob, avg_logprob, compression_ratio) instead of a blocklist.
- **Language routing**: `lang.py` detects CJK characters to reject unsupported scripts; everything else → Russian (single-language MVP).
- **Transcript forwarding**: `transcript.py` debounces user STT fragments (1s) before sending to the web UI via LiveKit data channel. Assistant responses are buffered between LLMFullResponseStart/End frames.
- **Rate limiting**: slowapi on API endpoints. When behind Caddy, `_real_ip()` reads X-Forwarded-For from trusted Docker bridge IPs only.
- **CORS**: wildcard in dev, domain-derived in prod (set via `CORS_ALLOWED_ORIGINS`).
- **Secrets flow (prod)**: Terraform → SSM Parameter Store → instance IAM role → `user_data.sh` fetches at boot → writes `infra/.env`.

## Linting

- **ruff**: line-length 100, target py312. Runs in pre-commit (local) and CI.
- **gitleaks**: secret scanning in pre-commit (local).
- CI job `Lint Python (services/agent)` runs `ruff check` from `services/agent/` with `--config pyproject.toml`.

## Conventions

- Commit format: `type(scope): description` — types: feat, fix, refactor, docs, test, chore, perf, ci. Scopes: api, agent, web, infra, ci.
- Python: type hints on all signatures, Pydantic for settings/validation, loguru with lazy `{name}` placeholders (not f-strings) for logger calls.
- Docker images: multi-stage, non-root (UID 1001 `appuser`), BuildKit cache mounts for uv/pip.
- Env vars validated at startup — `require_env()` in agent, Pydantic `Field(min_length=...)` in API settings.

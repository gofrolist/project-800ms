# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time voice assistant: browser ↔ LiveKit (WebRTC) ↔ Pipecat agent (VAD → GigaAM STT → LLM → TTS) with a FastAPI token-minting backend. Target: first audio out <800ms after end-of-speech.

## Architecture

```
Browser (React + LiveKit SDK)  ──WebRTC──►  LiveKit SFU (:7880)
       │                                        │
       │ POST /sessions                   Pipecat Agent (:8001)
       ▼                                   ├─ Silero VAD (CPU)
FastAPI (:8000) ──POST /dispatch──►        ├─ GigaAM-v3 STT (GPU, shared)
  mint LiveKit JWT                         ├─ LLM → vLLM or Groq/OpenAI
  dispatch agent to room                   └─ Piper TTS (CPU)
```

Four deployable units: `apps/api` (FastAPI), `services/agent` (Pipecat dispatcher), `apps/web` (React SPA), `infra/` (Docker Compose + Caddy + Terraform for AWS GPU spot).

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

# One-time (and after any services/agent dep change): build the heavy
# agent BASE image. ~5–7 GB, dominated by CUDA + PyTorch + faster-whisper.
docker build \
  -f services/agent/Dockerfile.base \
  -t project-800ms-agent-base:local \
  services/agent/

# The app build is a thin COPY of source on top of the base — seconds.
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d --build
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
- **GigaAM hallucination filter**: `gigaam_stt.py` drops segments below a duration floor (300ms) or token-count floor (2 tokens) — GigaAM doesn't expose Whisper-style per-segment confidence signals.
- **Language routing**: `lang.py` detects CJK characters to reject unsupported scripts; everything else → Russian (single-language MVP).
- **Transcript forwarding**: `transcript.py` debounces user STT fragments (1s) before sending to the web UI via LiveKit data channel. Assistant responses are buffered between LLMFullResponseStart/End frames.
- **Rate limiting**: token-bucket limiter in `apps/api/rate_limit.py` with separate caches for tenant (`_tenant_buckets`) and IP (`_ip_buckets`) keys. `enforce_tenant_rate_limit` protects authenticated `/v1/*` routes; `enforce_admin_ip_rate_limit` and `enforce_webhook_ip_rate_limit` protect the unauth admin and webhook surfaces. `_real_ip()` trusts X-Forwarded-For only when the TCP peer falls inside `settings.trusted_proxy_cidrs` (default RFC1918 172.16.0.0/12, configurable via env).
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
- Agent image split: `services/agent/Dockerfile.base` carries CUDA + Python deps (rebuilt only when `pyproject.toml` / `uv.lock` change); `services/agent/Dockerfile` is a thin `FROM ${BASE_IMAGE}` + `COPY . /app` layer on top. Model weights are NOT baked — the `hf_cache_agent` named volume persists HF downloads across restarts.
- Env vars validated at startup — `require_env()` in agent, Pydantic `Field(min_length=...)` in API settings.

## Documented Solutions

`docs/solutions/` — solutions to past problems (security issues, build errors, runtime issues, best practices), organized by category with YAML frontmatter (`title`, `module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan:
[specs/002-helper-guide-npc/plan.md](specs/002-helper-guide-npc/plan.md)

Companion artifacts in the same directory:
- [spec.md](specs/002-helper-guide-npc/spec.md) — feature requirements
- [research.md](specs/002-helper-guide-npc/research.md) — resolved technical decisions
- [data-model.md](specs/002-helper-guide-npc/data-model.md) — schema and state transitions
- [contracts/](specs/002-helper-guide-npc/contracts/) — retriever API + ingestion CLI
- [quickstart.md](specs/002-helper-guide-npc/quickstart.md) — dev bring-up

Constitution: [.specify/memory/constitution.md](.specify/memory/constitution.md) — v1.0.0.
Prior feature: [specs/001-voice-assistant-core/spec.md](specs/001-voice-assistant-core/spec.md)
(voice platform; this feature depends on it).
<!-- SPECKIT END -->

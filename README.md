# project-800ms

Self-hosted, low-latency voice assistant. Target: ≤800ms mic-to-speaker round trip.
No external API dependencies.

## Stack

```
Browser (React + LiveKit SDK)
   │  WebRTC
   ▼
LiveKit (SFU)  ◀──▶  Agent worker (Pipecat)
                       │   ├─ Silero VAD         (CPU, in-process)
                       │   ├─ Faster-Whisper     (GPU, in-process)
                       │   ├─ vLLM client ───────┼──▶ vLLM (separate GPU container)
                       │   └─ Piper TTS          (CPU, in-process)
                       ▼
                FastAPI (auth, sessions, LiveKit tokens)
                       │
                Postgres + Redis
```

See [plan](~/.claude/plans/inherited-sparking-torvalds.md) for full architecture.

## Prerequisites

- Docker + Compose v2
- NVIDIA Container Toolkit installed and configured
- `nvidia-smi` works on the host
- GPU with ≥16GB VRAM (RTX 5080 / L4 / A10G / L40S)

## Bring-up

```bash
# 1. Configure
cp infra/.env.example infra/.env
# edit infra/.env — set POSTGRES_PASSWORD and LIVEKIT_API_SECRET

# 2. Start
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d

# 3. Watch the slow one (vLLM downloads ~5GB on first run)
docker compose -f infra/docker-compose.yml logs -f vllm
# Wait for:  "Uvicorn running on http://0.0.0.0:8000"
```

## Verify

```bash
# API health
curl http://localhost:8000/health
# {"status":"ok"}

# vLLM — lists the served model
curl http://localhost:8001/v1/models

# vLLM — actual inference round-trip
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Привет! Скажи одну фразу."}],
    "max_tokens": 64
  }'

# LiveKit signaling
curl -i http://localhost:7880
# HTTP/1.1 404 — that's fine, means the server is up
```

### GPU check

```bash
# During vLLM startup, watch VRAM fill:
nvidia-smi
# Expect ~8-9GB used by the vllm container after load.
# Faster-Whisper will add another ~2GB once the agent starts pulling audio.
```

## Layout

```
apps/
  api/        FastAPI — auth, sessions, LiveKit token minting
  web/        React frontend (next step)
services/
  agent/      Pipecat worker — joins LiveKit rooms as a bot; Piper TTS runs here
  llm/        vLLM launch notes (image + args only, no custom code)
infra/
  docker-compose.yml   Full stack
  livekit.yaml         LiveKit server config
  .env.example         Secrets template
```

## End-to-end flow

1. `docker compose up` → `api`, `agent`, `livekit`, `vllm`, `postgres`, `redis` come up
2. Agent auto-joins the `demo` room and waits
3. Open `http://localhost:5173` (web) and click **Start call**
4. Browser: `POST /sessions` → gets `{url, token, room}`
5. Browser joins the same `demo` room via LiveKit → mic published, agent audio subscribed
6. Speak (EN or RU). Pipecat runs VAD → Whisper → Qwen2.5 → Piper. Target round trip ≤ 1s.

## Running the web client (dev)

```bash
cd apps/web
bun install       # https://bun.sh
bun run dev       # http://localhost:5173
```

The backend stack must already be up (`docker compose -f infra/docker-compose.yml up`).

## Development setup

After cloning, install [pre-commit](https://pre-commit.com/) and enable the hooks once:

```bash
# macOS:    brew install pre-commit
# pip:      pipx install pre-commit
# Then, in the repo root:
pre-commit install
```

This wires up file-hygiene checks, [`ruff`](https://docs.astral.sh/ruff/) lint+format,
[`gitleaks`](https://github.com/gitleaks/gitleaks) secret scanning,
[`actionlint`](https://github.com/rhysd/actionlint), and JSON-Schema validation
of `.github/workflows/` and `dependabot.yml`. The same checks run in CI, so
catching them locally saves a round trip.

To run the hooks against the whole tree (e.g. after pulling new commits):

```bash
pre-commit run --all-files
```

The Python services use [`uv`](https://docs.astral.sh/uv/) for dependency management:

```bash
cd apps/api && uv sync           # install deps
cd apps/api && uv run pytest     # run tests
```

Same for `services/agent/`.

## Current status

End-to-end MVP: transport + agent + API + minimal web client. Single-tenant
(one shared `demo` room). Next up: per-call dispatch, auth/users, observability,
and a real frontend.

## Teardown

```bash
docker compose -f infra/docker-compose.yml down           # stop
docker compose -f infra/docker-compose.yml down -v        # stop + wipe volumes (postgres, model cache)
```

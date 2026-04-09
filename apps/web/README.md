# web

Minimal React client: one button, LiveKit voice room, agent audio visualizer.

```bash
cd apps/web
bun install          # https://bun.sh
bun run dev          # http://localhost:5173
```

The dev server talks to the API at `VITE_API_URL` (default `http://localhost:8000`).
Make sure the backend stack is up (`docker compose -f infra/docker-compose.yml up`).

## Flow

1. Click **Start call**
2. Browser: `POST /sessions` → `{url, token, room, identity}`
3. `<LiveKitRoom>` connects via WebRTC, publishes mic, subscribes to agent audio
4. The agent (already sitting in the `demo` room) greets you
5. Speak — Pipecat runs VAD → Whisper → Qwen2.5 → Piper, round-trip target ≤1s

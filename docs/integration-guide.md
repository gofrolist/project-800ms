# Game client integration guide

How a game (or any real-time voice client) connects to the voice-assistant service end-to-end.

## The short version

Your client does **not** send audio through our REST API. The REST API only mints a room and a short-lived token. Audio flows over **WebRTC** directly to the LiveKit SFU using the LiveKit client SDK. The voice agent joins the same room and appears to you as another participant.

```
Game client ──HTTPS──► API                POST /v1/sessions → { url, token, room }
     │
     │ ──WebRTC (SRTP/DTLS over UDP, WSS fallback)──►  LiveKit SFU
     │                                                       ▲
     │                                                       │ joins same room
     │                                                   Voice agent
     │                                                   (VAD → STT → LLM → TTS)
     │
     │ ──HTTPS──► DELETE /v1/sessions/{room}   (hang up)
```

Four things you need to integrate:

1. Get an API key from ops.
2. `POST /v1/sessions` — receive `url`, `token`, `room`, `identity`.
3. Connect with a LiveKit client SDK using those values.
4. `DELETE /v1/sessions/{room}` when the user hangs up.

---

## Authentication

Every `/v1/*` call requires an `X-API-Key` header. Keys are tenant-scoped.

- Rate limit: **60 requests/minute per tenant** (default; configurable per-tenant).
- Revocation propagates within ~60s (the API caches resolved identities in a TTL cache).
- Raw keys are never logged. Support tickets reference the 8-char key prefix.

Errors use a unified envelope:

```json
{
  "error": {
    "code": "unauthenticated",
    "message": "Missing or malformed X-API-Key header",
    "request_id": "01HQXZ3Y4A5TGJ6GGC8CVKWE0N"
  }
}
```

`request_id` is also returned in the `X-Request-ID` response header — include it in support tickets.

---

## Endpoints

### `GET /v1/engines` — discover available TTS engines (optional)

Reports which TTS engines the agent has preloaded, which is the default, and how to shape the `voice` field for each.

```http
GET /v1/engines
X-API-Key: sk_live_xxx
```

```json
{
  "engines": [
    {"id":"piper",  "available":true,  "default":true,  "voice_format":"piper_voice_name",  "label":"Piper — CPU, baseline"},
    {"id":"silero", "available":true,  "default":false, "voice_format":"silero_speaker_id", "label":"Silero v5 — GPU, RU"},
    {"id":"qwen3",  "available":false, "default":false, "voice_format":"openai_or_clone",   "label":"Qwen3-TTS sidecar"},
    {"id":"xtts",   "available":true,  "default":false, "voice_format":"clone_only",        "label":"Coqui XTTS v2"}
  ]
}
```

| `voice_format`       | How to shape `voice` |
|----------------------|----------------------|
| `piper_voice_name`   | HuggingFace voice-pack name, e.g. `ru_RU-denis-medium` |
| `silero_speaker_id`  | Speaker id in the v5_cis_base model, e.g. `ru_ekaterina` |
| `openai_or_clone`    | OpenAI voice name (`alloy`, `echo`, …) **or** `clone:<profile>` |
| `clone_only`         | Always `clone:<profile>` — XTTS is zero-shot cloning, no presets |

If you always use the default engine, you can skip this call.

Errors: `503` if the agent is unreachable. Retry later.

---

### `POST /v1/sessions` — open a voice session

Creates a LiveKit room, dispatches the agent into it, returns credentials the client uses to join.

```http
POST /v1/sessions
X-API-Key: sk_live_xxx
Content-Type: application/json
```

Request body — every field is optional:

| Field        | Type              | Purpose |
|--------------|-------------------|---------|
| `user_id`    | `string ≤128`     | Stable caller identifier in your system. Used as LiveKit identity. |
| `npc_id`     | `string ≤128`     | Which NPC/persona this session targets. |
| `persona`    | `object`          | Persona JSON — system prompt, backstory, knobs. |
| `voice`      | `string ≤64`      | Engine-specific voice id (see `/v1/engines`). |
| `language`   | `string ≤16`      | BCP-47 hint for STT + LLM output. Today: `ru` (single-language MVP). |
| `llm_model`  | `string ≤128`     | Override LLM model. |
| `context`    | `object`          | Arbitrary game-state context forwarded to the agent. |
| `tts_engine` | `piper\|silero\|qwen3\|xtts` | Engine selection. Falls back to deploy default. |

Unknown fields are rejected (`extra="forbid"`).

Validation rule to remember: `tts_engine="xtts"` requires `voice="clone:<profile>"`. XTTS has no preset voices; a mismatch fails fast with `422` before the agent is dispatched.

Response — `201 Created`:

```json
{
  "session_id": "e217f8e3-a5e3-433b-8d1a-…",
  "room":       "room-3f8a1c92",
  "identity":   "player-42",
  "token":      "eyJhbGciOi…",
  "url":        "wss://livekit.coastalai.ai"
}
```

- `url` — LiveKit SFU WebSocket URL (pass to the SDK's `connect()`).
- `token` — LiveKit JWT scoped to this one room, TTL 15 min.
- `room` — room name (use this for subsequent `GET`/`DELETE` calls).
- `identity` — echoes `user_id` if supplied, else an auto-generated `user-<hex>`.

Error codes:

| Status | `code`                | Meaning |
|--------|-----------------------|---------|
| 401    | `unauthenticated`     | Missing/bad `X-API-Key`. |
| 403    | `forbidden`           | Key revoked or tenant suspended. |
| 422    | —                     | Body validation failure (e.g. xtts + non-clone voice). |
| 429    | `rate_limited`        | >60/min for this tenant. |
| 503    | `agent_unavailable`   | Agent down; DB row rolled back, retry safe. |

---

### `GET /v1/sessions/{room}` — read session state

```json
{
  "session_id": "e217f8e3-…",
  "room": "room-3f8a1c92",
  "identity": "player-42",
  "user_id": "player-42",
  "npc_id": "blacksmith",
  "persona": {...},
  "voice": "ru_RU-denis-medium",
  "language": "ru",
  "llm_model": "qwen-7b",
  "context": {...},
  "status": "active",
  "created_at": "2026-04-23T12:00:00+00:00",
  "started_at": "2026-04-23T12:00:00+00:00",
  "ended_at":   null,
  "audio_seconds": null
}
```

`status` lifecycle: `pending → active → ended` (or `failed` if the pipeline errored). `404` for sessions owned by other tenants — existence is deliberately not distinguishable across tenants.

---

### `GET /v1/sessions/{room}/transcripts` — read persisted utterances

Up to 1000 entries, ascending by `created_at`.

```json
{
  "transcripts": [
    {"id":"…","role":"user",     "text":"Привет","started_at":"…","ended_at":"…","created_at":"…"},
    {"id":"…","role":"assistant","text":"Здравствуй, путник!","started_at":null,"ended_at":null,"created_at":"…"}
  ],
  "count": 2
}
```

**Security note:** `text` is raw STT output. Escape before rendering in HTML contexts (e.g. DOMPurify).

---

### `DELETE /v1/sessions/{room}` — end a voice session

User-initiated hang-up. Tears down the LiveKit room (kicks caller + agent), marks `status=ended`, records `audio_seconds`.

- **Idempotent.** Calling twice returns the existing details.
- Returns `409 conflict` on `status=pending` — retry once the POST commits (narrow window).
- Terminal `failed` status is preserved, not overwritten with `ended`.
- Safe against LiveKit flakes — DB close still commits even if the LiveKit API call fails.

If the client skips DELETE (tab close, crash), the agent's `on_participant_left` handler triggers teardown automatically. DELETE is tidier and frees the GPU sooner.

---

## Connecting the voice channel (WebRTC via LiveKit)

Once you have `{url, token, room, identity}` the REST API is done. Use a LiveKit client SDK:

| Platform          | SDK |
|-------------------|-----|
| Web / React       | [`livekit-client`](https://github.com/livekit/client-sdk-js), [`@livekit/components-react`](https://github.com/livekit/components-js) |
| Unity             | [`livekit-unity`](https://github.com/livekit/client-sdk-unity) |
| Unreal            | [`livekit-unreal`](https://github.com/livekit/client-sdk-unreal) |
| iOS / macOS       | [`client-sdk-swift`](https://github.com/livekit/client-sdk-swift) |
| Android / Kotlin  | [`client-sdk-android`](https://github.com/livekit/client-sdk-android) |
| Flutter           | [`client-sdk-flutter`](https://github.com/livekit/client-sdk-flutter) |
| Node / bots       | [`client-sdk-js`](https://github.com/livekit/client-sdk-js) + wrtc, or [`agents`](https://github.com/livekit/agents) |

Minimum connect flow (JS, but the shape is identical on every SDK):

```ts
import { Room } from 'livekit-client';

const room = new Room({ adaptiveStream: true, dynacast: true });
await room.connect(session.url, session.token);     // WebRTC handshake
await room.localParticipant.setMicrophoneEnabled(true);  // publish mic
// Remote audio from `agent-bot-<room>` auto-plays via the SDK.
```

Reference implementation: [`apps/web/src/App.tsx`](../apps/web/src/App.tsx) uses `<LiveKitRoom serverUrl={session.url} token={session.token} audio />`.

### What flows over the wire

- **Uplink**: mic audio, Opus-encoded, SRTP over DTLS. ICE/UDP preferred, WSS/TCP fallback through the SFU.
- **Downlink**: one Opus audio track published by participant `agent-bot-<room>` (the TTS output).
- **Data channel** (JSON text, reliable): transcripts and error frames from the agent.

Data-channel payloads:

```ts
// Fired for user utterances (debounced 1s after end-of-speech) and
// assistant responses (buffered between LLMFullResponseStart/End frames).
{ type: "transcript", role: "user" | "assistant", text: string }

// Soft/fatal error surfaces — render as a banner, not a transcript row.
{ type: "error", error: string, fatal: boolean }
```

Subscribe via the SDK's data-channel API (`room.on('dataReceived', ...)` or `useDataChannel` in React). Unknown `type` values should be ignored — the protocol will grow.

### What the agent does inside the room

You don't implement this — it's automatic per session — but for the mental model:

1. Silero VAD decides when the caller's speech ends.
2. GigaAM-v3 transcribes. Hallucination filter drops utterances <300 ms or <2 tokens.
3. LLM (vLLM local, or Groq/OpenAI remote) generates a reply using `persona` + `context`.
4. TTS (`tts_engine`) synthesizes the reply as an Opus track.
5. Target: first audio out **<800 ms** after end-of-speech.

Rooms are isolated. Two callers can never cross-talk even if you reuse `user_id`.

---

## End-to-end example (curl + browser)

```bash
# 1. Open session
curl -sX POST https://api.coastalai.ai/v1/sessions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "user_id":   "player-42",
        "npc_id":    "blacksmith",
        "voice":     "ru_RU-denis-medium",
        "language":  "ru",
        "tts_engine":"piper"
      }'
# → {"session_id":"…","room":"room-3f8a1c92","identity":"player-42",
#    "token":"eyJ…","url":"wss://livekit.coastalai.ai"}
```

```ts
// 2. Connect the audio (browser/game)
import { Room } from 'livekit-client';
const room = new Room();
await room.connect(url, token);
await room.localParticipant.setMicrophoneEnabled(true);

room.on('dataReceived', (payload, participant) => {
  const msg = JSON.parse(new TextDecoder().decode(payload));
  if (msg.type === 'transcript') updateCaptions(msg.role, msg.text);
  else if (msg.type === 'error') showBanner(msg.error, msg.fatal);
});
```

```bash
# 3. Hang up when the user ends the call
curl -sX DELETE https://api.coastalai.ai/v1/sessions/room-3f8a1c92 \
  -H "X-API-Key: $API_KEY"
```

---

## Operational notes

- **One room = one caller.** Don't reuse rooms or multiplex callers.
- **Token TTL is 15 min.** Mint tokens per call; don't pre-mint and shelf them.
- **Tenant rate limit: 60 req/min.** Sessions are long-lived, so this is rarely a concern — but bursty retry loops will hit it.
- **Always DELETE on user-initiated hang-up.** Automatic teardown on disconnect works but is slower to reclaim the GPU.
- **Stick to the SDK.** Raw SDP against LiveKit works but isn't worth the maintenance — every major platform has an SDK.
- **CORS (browser clients only).** Prod has an explicit origin allowlist. If a browser gets a CORS error, your origin isn't on the list — contact ops.
- **`request_id` in every response.** Forward it to support when reporting issues.

---

## Full OpenAPI spec

Every field, every error shape, always up to date:

- Raw spec: `GET /openapi.json`
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- Scalar: `/reference`

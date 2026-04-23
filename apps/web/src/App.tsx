import { useCallback, useRef, useState, useEffect } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  StartAudio,
  BarVisualizer,
  useVoiceAssistant,
  useDataChannel,
} from "@livekit/components-react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

interface VoiceOption {
  id: string;
  label: string;
}

// Non-empty tuple — `voices[0].id` is read unguarded at state init time.
// Enforcing non-emptiness in the type means an engine with zero voices
// fails to compile instead of throwing at App mount.
type NonEmptyVoices = readonly [VoiceOption, ...VoiceOption[]];

interface EngineDescriptor {
  // id is not constrained to a specific union here — TtsEngine is
  // derived from TTS_ENGINES below, so constraining here would be a
  // circular dependency. The `satisfies` at the array site keeps the
  // id values narrow during authoring.
  id: string;
  label: string;
  sub: string;
  voices: NonEmptyVoices;
}

// Per-engine voice catalogs. Voice ids must match what the engine expects:
// - Piper: HuggingFace voice-pack name (auto-downloaded on first use)
// - Silero: speaker id within the v5_cis_base model (see
//   services/agent/silero_tts.py SileroSettings.speaker)
// - Qwen3: clone:<profile> identifier resolved via voice_library/profiles/
//   (baked into the sidecar image)
// - XTTS: clone:<profile> identifier resolved against the agent's local
//   voice_library mount (same directory shape as Qwen3; see
//   services/agent/xtts_tts.py _resolve_voice_profile)
//
// Curated to a few options per engine so the dropdown stays scannable.
// Add more here as you commission new voices on the backend.
//
// `satisfies` validates the shape without widening away the literal
// types, so `TtsEngine` below can be inferred as the narrow string-
// literal union of actual `id` values rather than `string`.
const TTS_ENGINES = [
  {
    id: "piper",
    label: "Piper",
    sub: "CPU · baseline",
    voices: [
      { id: "ru_RU-denis-medium", label: "Denis (M)" },
      { id: "ru_RU-dmitri-medium", label: "Dmitri (M)" },
      { id: "ru_RU-ruslan-medium", label: "Ruslan (M)" },
      { id: "ru_RU-irina-medium", label: "Irina (F)" },
    ],
  },
  {
    id: "silero",
    label: "Silero",
    sub: "GPU · v5 RU",
    voices: [
      { id: "ru_zhadyra", label: "Zhadyra (F)" },
      { id: "ru_ekaterina", label: "Ekaterina (F)" },
      { id: "ru_vika", label: "Vika (F)" },
      { id: "ru_oksana", label: "Oksana (F)" },
      { id: "ru_dmitriy", label: "Dmitriy (M)" },
      { id: "ru_eduard", label: "Eduard (M)" },
      { id: "ru_alexandr", label: "Aleksandr (M)" },
      { id: "ru_roman", label: "Roman (M)" },
    ],
  },
  {
    id: "qwen3",
    label: "Qwen3",
    sub: "GPU · cloned",
    voices: [{ id: "clone:demo-ru", label: "Cloned voice" }],
  },
  {
    id: "xtts",
    label: "XTTS v2",
    sub: "GPU · cloned",
    voices: [{ id: "clone:demo-ru", label: "Cloned voice" }],
  },
] as const satisfies readonly EngineDescriptor[];

// Single source of truth for the engine union: derived from the
// narrow tuple that `as const` + `satisfies` preserves. Adding an entry
// above automatically extends `TtsEngine`; removing one narrows it. No
// second place to keep in sync.
type TtsEngine = (typeof TTS_ENGINES)[number]["id"];

interface Session {
  session_id: string;
  url: string;
  token: string;
  room: string;
  identity: string;
  tts_engine: TtsEngine;
  voice: string;
}

interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    request_id?: string;
  };
}

type Status = "idle" | "connecting" | "live" | "error";

interface Message {
  role: "user" | "assistant";
  text: string;
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ErrorEnvelope;
    if (body?.error?.message) {
      return `${body.error.code}: ${body.error.message}`;
    }
  } catch {
    // Fall through to the generic status-based message below.
  }
  return `API ${res.status}`;
}

async function createSession(ttsEngine: TtsEngine, voice: string): Promise<Session> {
  if (!API_KEY) {
    throw new Error(
      "VITE_API_KEY is not configured. Set it at build time or via the web container's API_KEY env.",
    );
  }
  const res = await fetch(`${API_URL}/v1/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify({ tts_engine: ttsEngine, voice }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  const data = (await res.json()) as Omit<Session, "tts_engine" | "voice">;
  // /v1/sessions doesn't echo back tts_engine / voice; carry the
  // client's choice through so the CallView can surface them.
  return { ...data, tts_engine: ttsEngine, voice };
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [pendingEngine, setPendingEngine] = useState<TtsEngine | null>(null);
  const [error, setError] = useState<string | null>(null);
  // One selected voice per engine. Initialized to each engine's first
  // voice so the dropdowns always have a sensible default.
  //
  // Built via reduce with an explicit accumulator type so TypeScript
  // verifies completeness against TtsEngine. An Object.fromEntries +
  // `as Record<TtsEngine, string>` cast would silently paper over a
  // future engine added to the union but not to TTS_ENGINES; with
  // TtsEngine now derived from TTS_ENGINES this is less likely, but the
  // reduce pattern keeps the invariant visible at the call site.
  const [selectedVoice, setSelectedVoice] = useState<Record<TtsEngine, string>>(() =>
    TTS_ENGINES.reduce(
      (acc, e) => {
        acc[e.id] = e.voices[0].id;
        return acc;
      },
      {} as Record<TtsEngine, string>,
    ),
  );

  const startCall = useCallback(
    async (engine: TtsEngine, voice: string) => {
      setStatus("connecting");
      setPendingEngine(engine);
      setError(null);
      try {
        setSession(await createSession(engine, voice));
        setStatus("live");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      } finally {
        setPendingEngine(null);
      }
    },
    [],
  );

  const endCall = useCallback(() => {
    setSession(null);
    setStatus("idle");
  }, []);

  if (session) {
    return (
      <LiveKitRoom
        serverUrl={session.url}
        token={session.token}
        connect
        audio
        video={false}
        onDisconnected={endCall}
        options={{
          publishDefaults: { red: false, dtx: false },
        }}
      >
        <RoomAudioRenderer />
        <StartAudio label="Click to enable audio" />
        <CallView
          onEnd={endCall}
          identity={session.identity}
          ttsEngine={session.tts_engine}
          voice={session.voice}
        />
      </LiveKitRoom>
    );
  }

  return (
    <div className="app">
      <h1>project-800ms</h1>
      <p className="subtitle">Pick a TTS engine + voice to start a call.</p>
      <div className="engine-picker">
        {TTS_ENGINES.map((engine) => {
          const isPending = pendingEngine === engine.id && status === "connecting";
          const voice = selectedVoice[engine.id];
          const singleVoice = engine.voices.length === 1;
          return (
            <div key={engine.id} className="engine-card">
              <div className="engine-card-head">
                <span className="engine-label">{engine.label}</span>
                <span className="engine-sub">{engine.sub}</span>
              </div>
              <select
                className="engine-voice"
                value={voice}
                onChange={(e) =>
                  setSelectedVoice((prev) => ({ ...prev, [engine.id]: e.target.value }))
                }
                disabled={status === "connecting" || singleVoice}
                aria-label={`${engine.label} voice`}
              >
                {engine.voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </select>
              <button
                className="engine-start"
                onClick={() => startCall(engine.id, voice)}
                disabled={status === "connecting"}
              >
                {isPending ? "Connecting…" : "Start"}
              </button>
            </div>
          );
        })}
      </div>
      {error && <div className="status error">Error: {error}</div>}
    </div>
  );
}

interface CallViewProps {
  onEnd: () => void;
  identity: string;
  ttsEngine: TtsEngine;
  voice: string;
}

function CallView({ onEnd, identity, ttsEngine, voice }: CallViewProps) {
  const { state, audioTrack } = useVoiceAssistant();
  const [messages, setMessages] = useState<Message[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useDataChannel((msg) => {
    try {
      const text = new TextDecoder().decode(msg.payload);
      const parsed = JSON.parse(text) as Message;
      if (parsed.role && parsed.text) {
        setMessages((prev) => [...prev, parsed]);
      }
    } catch {
      // ignore non-JSON data messages
    }
  });

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  return (
    <div className="app call-layout">
      <h1>In call</h1>
      <div style={{ width: 240, height: 120, margin: "0 auto 1rem" }}>
        <BarVisualizer state={state} trackRef={audioTrack} barCount={12} />
      </div>
      <div className="transcript" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <span className="msg-role">{m.role === "user" ? "Вы" : "Ассистент"}</span>
            <span className="msg-text">{m.text}</span>
          </div>
        ))}
      </div>
      <div className="status">
        you: <code>{identity}</code> · tts: <code>{ttsEngine}</code> · voice:{" "}
        <code>{voice}</code> · assistant: <code>{state}</code>
      </div>
      <button
        className="start-btn"
        style={{ marginTop: "1rem", background: "#dc2626" }}
        onClick={onEnd}
      >
        End call
      </button>
    </div>
  );
}

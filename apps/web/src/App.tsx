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

type TtsEngine = "piper" | "silero" | "qwen3";

const TTS_ENGINES: readonly { id: TtsEngine; label: string; sub: string }[] = [
  { id: "piper", label: "Piper", sub: "CPU · baseline" },
  { id: "silero", label: "Silero", sub: "GPU · v5 RU" },
  { id: "qwen3", label: "Qwen3", sub: "GPU · 0.6B" },
] as const;

interface Session {
  session_id: string;
  url: string;
  token: string;
  room: string;
  identity: string;
  tts_engine: TtsEngine;
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

async function createSession(ttsEngine: TtsEngine): Promise<Session> {
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
    body: JSON.stringify({ tts_engine: ttsEngine }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  const data = (await res.json()) as Omit<Session, "tts_engine">;
  // /v1/sessions doesn't echo back tts_engine; carry the client's choice
  // through so the CallView can surface which engine is speaking.
  return { ...data, tts_engine: ttsEngine };
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [pendingEngine, setPendingEngine] = useState<TtsEngine | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startCall = useCallback(async (engine: TtsEngine) => {
    setStatus("connecting");
    setPendingEngine(engine);
    setError(null);
    try {
      setSession(await createSession(engine));
      setStatus("live");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    } finally {
      setPendingEngine(null);
    }
  }, []);

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
        <CallView onEnd={endCall} identity={session.identity} ttsEngine={session.tts_engine} />
      </LiveKitRoom>
    );
  }

  return (
    <div className="app">
      <h1>project-800ms</h1>
      <p className="subtitle">Pick a TTS engine to start a call.</p>
      <div className="engine-picker">
        {TTS_ENGINES.map((engine) => {
          const isPending = pendingEngine === engine.id && status === "connecting";
          return (
            <button
              key={engine.id}
              className="engine-btn"
              onClick={() => startCall(engine.id)}
              disabled={status === "connecting"}
            >
              <span className="engine-label">{isPending ? "Connecting…" : engine.label}</span>
              <span className="engine-sub">{engine.sub}</span>
            </button>
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
}

function CallView({ onEnd, identity, ttsEngine }: CallViewProps) {
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
        you: <code>{identity}</code> · tts: <code>{ttsEngine}</code> · assistant:{" "}
        <code>{state}</code>
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

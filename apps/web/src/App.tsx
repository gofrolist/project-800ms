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

interface Session {
  session_id: string;
  url: string;
  token: string;
  room: string;
  identity: string;
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

async function createSession(): Promise<Session> {
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
    // Empty body uses tenant / agent defaults. Add user_id / npc_id /
    // persona here when wiring this SPA into a game that has a current
    // character context.
    body: "{}",
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as Session;
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const startCall = useCallback(async () => {
    setStatus("connecting");
    setError(null);
    try {
      setSession(await createSession());
      setStatus("live");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
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
        <CallView onEnd={endCall} identity={session.identity} />
      </LiveKitRoom>
    );
  }

  return (
    <div className="app">
      <h1>project-800ms</h1>
      <button className="start-btn" onClick={startCall} disabled={status === "connecting"}>
        {status === "connecting" ? "Connecting…" : "Start call"}
      </button>
      {error && <div className="status error">Error: {error}</div>}
    </div>
  );
}

function CallView({ onEnd, identity }: { onEnd: () => void; identity: string }) {
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
        you: <code>{identity}</code> · assistant: <code>{state}</code>
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

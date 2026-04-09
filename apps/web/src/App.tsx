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

type Session = { url: string; token: string; room: string; identity: string };

type Status = "idle" | "connecting" | "live" | "error";

type Message = { role: "user" | "assistant"; text: string };

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const startCall = useCallback(async () => {
    setStatus("connecting");
    setError(null);
    try {
      const res = await fetch(`${API_URL}/sessions`, { method: "POST" });
      if (!res.ok) throw new Error(`API ${res.status}`);
      setSession((await res.json()) as Session);
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
      <button className="start-btn" style={{ marginTop: "1rem", background: "#dc2626" }} onClick={onEnd}>
        End call
      </button>
    </div>
  );
}

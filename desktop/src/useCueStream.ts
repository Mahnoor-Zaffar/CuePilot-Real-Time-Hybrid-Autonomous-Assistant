/**
 * useCueStream — the renderer's only view of the server.
 *
 * Responsibilities that belong on the client:
 *   • handshake + session resume (session id survives reloads in localStorage)
 *   • jittered exponential reconnect, so a server restart is invisible
 *   • requestAnimationFrame delta batching: the socket may emit up to ~60
 *     frames/s; we paint at most once per animation frame
 *   • heartbeat: answer server pings, send our own for an RTT readout
 *   • per-turn cancellation and tool-call assembly
 *   • raw PCM mic capture -> 16-bit base64 frames
 *
 * Everything is expressed as plain wire frames; no server types leak in.
 */
import { useCallback, useEffect, useRef, useState } from "react";

// ── wire types (mirror app/ws/protocol.py) ────────────────────────────────
export type RetrievalMode = "hybrid" | "bm25" | "dense";

export interface PromptOptions {
  top_k?: number;
  mode: RetrievalMode;
  tools: boolean;
  temperature?: number;
  max_tokens?: number;
}

export interface SourceHit {
  id: string;
  doc_id: string;
  title: string;
  content?: string;
  score: number;
  meta?: Record<string, unknown>;
  explain?: {
    ranks: Record<string, number>;
    contributions: Record<string, number>;
    raw_scores: Record<string, number>;
  };
}

export interface RetrievalInfo {
  query: string;
  mode: string;
  kb_version: number;
  cache_hit: boolean;
  degraded: string[];
  embedder: string;
  timings_ms: Record<string, number>;
  candidates: Record<string, number>;
}

export interface ToolCallView {
  callId: string;
  name: string;
  arguments: unknown;
  status: "running" | "ok" | "error";
  latencyMs?: number;
  result?: unknown;
}

export interface TurnMetrics {
  finishReason?: string;
  ttftMs?: number | null;
  totalMs?: number | null;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  model?: string;
  toolIterations?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: number;
  streaming?: boolean;
  sources?: SourceHit[];
  retrieval?: RetrievalInfo;
  tools?: ToolCallView[];
  metrics?: TurnMetrics;
  error?: { code: string; message: string; retryable: boolean };
  aborted?: boolean;
  /** recovered from a streaming checkpoint after a crash; text may be partial */
  interrupted?: boolean;
  /** wall-clock prompt -> first painted delta, measured in the renderer */
  clientTtftMs?: number;
  /** server-side prompt -> first delta handed to the writer */
  serverTtftMs?: number | null;
}

export interface StreamStats {
  frames: number;
  deltaFrames: number;
  tokens: number;
  turns: number;
  rttMs: number | null;
  lastTtftMs: number | null;
  lastTotalMs: number | null;
}

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "closed";

export type AudioSource = "mic" | "system";

export interface AudioState {
  active: boolean;
  source: AudioSource | null;
  error: string | null;
  level: number;
  bufferedMs: number;
  transcribing: boolean;
  transcripts: string[];
}

type ServerFrame = { type: string } & Record<string, any>;

// ── constants ─────────────────────────────────────────────────────────────
const SESSION_KEY = "cuepilot.session_id";
const PROTOCOL_VERSION = 1;
const RECONNECT_BASE_MS = 250;
const RECONNECT_CAP_MS = 8_000;
const RECONNECT_MAX_ATTEMPT = 8;
const PING_INTERVAL_MS = 20_000;
const AUDIO_CHUNK_MS = 100;

const INITIAL_STATS: StreamStats = {
  frames: 0,
  deltaFrames: 0,
  tokens: 0,
  turns: 0,
  rttMs: null,
  lastTtftMs: null,
  lastTotalMs: null,
};

const INITIAL_AUDIO: AudioState = {
  active: false,
  source: null,
  error: null,
  level: 0,
  bufferedMs: 0,
  transcribing: false,
  transcripts: [],
};

const PCM_WORKLET = `
class PcmTap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("pcm-tap", PcmTap);
`;

// ── helpers ───────────────────────────────────────────────────────────────
function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function resolveSocketUrl(): string {
  const base = import.meta.env.VITE_WS_URL ?? "ws://127.0.0.1:8000/ws/stream";
  const token = import.meta.env.VITE_WS_TOKEN;
  if (!token) return base;
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}token=${encodeURIComponent(token)}`;
}

function int16ToBase64(samples: Int16Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let binary = "";
  const stride = 0x8000; // avoid blowing the argument limit on String.fromCharCode
  for (let i = 0; i < bytes.length; i += stride) {
    binary += String.fromCharCode(...bytes.subarray(i, i + stride));
  }
  return btoa(binary);
}

function readStoredSession(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function persistSession(sessionId: string | null): void {
  try {
    if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    /* storage can be disabled; session resume is best effort */
  }
}

interface AudioHandles {
  kind: AudioSource;
  ctx: AudioContext;
  stream: MediaStream;
  source: MediaStreamAudioSourceNode;
  node: AudioWorkletNode;
  sink: GainNode;
}

// ── hook ──────────────────────────────────────────────────────────────────
export function useCueStream() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [retryInMs, setRetryInMs] = useState(0);
  const [lastError, setLastError] = useState<{ code: string; message: string; retryable: boolean } | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(readStoredSession);
  const [lastClose, setLastClose] = useState<{ code: number; reason: string } | null>(null);
  const [model, setModel] = useState<string>("—");
  const [kbVersion, setKbVersion] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [stats, setStats] = useState<StreamStats>(INITIAL_STATS);
  const [audio, setAudio] = useState<AudioState>(INITIAL_AUDIO);

  const wsRef = useRef<WebSocket | null>(null);
  const connectRef = useRef<(() => void) | null>(null);
  const sessionIdRef = useRef<string | null>(sessionId);
  const attemptRef = useRef(0);
  const intentionalCloseRef = useRef(false);
  const activeTurnRef = useRef<string | null>(null);
  const pingTimerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<Map<string, string>>(new Map());
  const startedAtRef = useRef<Map<string, number>>(new Map());
  const clientTtftRef = useRef<Map<string, number>>(new Map());
  const firstDeltaRef = useRef<Set<string>>(new Set());
  const statsRef = useRef<StreamStats>({ ...INITIAL_STATS });
  const captureRef = useRef<AudioHandles | null>(null);
  const audioSeqRef = useRef(0);

  // ── stats / delta batching ──────────────────────────────────────────────
  const publishStats = useCallback(() => {
    setStats({ ...statsRef.current });
  }, []);

  const applyPending = useCallback(() => {
    rafRef.current = null;
    const pending = pendingRef.current;
    if (pending.size === 0) return;
    pendingRef.current = new Map();
    setMessages((prev) =>
      prev.map((message) => {
        const addition = pending.get(message.id);
        return addition ? { ...message, text: message.text + addition } : message;
      }),
    );
    publishStats();
  }, [publishStats]);

  const queueDelta = useCallback(
    (turnId: string, text: string) => {
      if (!text) return;
      const existing = pendingRef.current.get(turnId) ?? "";
      pendingRef.current.set(turnId, existing + text);
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(applyPending);
      }
    },
    [applyPending],
  );

  const flushPending = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    applyPending();
  }, [applyPending]);

  // ── heartbeat ───────────────────────────────────────────────────────────
  const stopPing = useCallback(() => {
    if (pingTimerRef.current !== null) {
      window.clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }
  }, []);

  const startPing = useCallback(() => {
    stopPing();
    pingTimerRef.current = window.setInterval(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
      }
    }, PING_INTERVAL_MS);
  }, [stopPing]);

  // ── frame dispatch ──────────────────────────────────────────────────────
  const handleFrame = useCallback(
    (raw: string) => {
      let frame: ServerFrame;
      try {
        frame = JSON.parse(raw) as ServerFrame;
      } catch {
        return;
      }
      if (!frame || typeof frame.type !== "string") return;
      statsRef.current.frames += 1;

      switch (frame.type) {
        case "ready": {
          attemptRef.current = 0;
          const nextSession = String(frame.session_id ?? "");
          sessionIdRef.current = nextSession || null;
          persistSession(nextSession || null);
          setSessionId(nextSession || null);
          setModel(String(frame.model ?? "—"));
          setKbVersion(typeof frame.kb_version === "number" ? frame.kb_version : null);
          setRetryInMs(0);
          setLastError(null);
          setStatus("open");
          startPing();
          publishStats();
          break;
        }

        case "history": {
          flushPending();
          pendingRef.current.clear();
          firstDeltaRef.current.clear();
          const now = Date.now();
          const restored: ChatMessage[] = Array.isArray(frame.messages)
            ? frame.messages
                .filter((item: any) => item && (item.role === "user" || item.role === "assistant"))
                .map((item: any, index: number) => ({
                  id: `history-${now}-${index}`,
                  role: item.role as "user" | "assistant",
                  text: String(item.content ?? ""),
                  createdAt: now + index,
                  interrupted: item.interrupted === true,
                }))
            : [];
          if (restored.length) setMessages(restored);
          break;
        }

        case "ack":
          break;

        case "retrieval": {
          const hits: SourceHit[] = Array.isArray(frame.hits) ? frame.hits : [];
          const info: RetrievalInfo = {
            query: String(frame.query ?? ""),
            mode: String(frame.mode ?? "hybrid"),
            kb_version: Number(frame.kb_version ?? 0),
            cache_hit: Boolean(frame.cache_hit),
            degraded: Array.isArray(frame.degraded) ? frame.degraded.map(String) : [],
            embedder: String(frame.embedder ?? ""),
            timings_ms: (frame.timings_ms ?? {}) as Record<string, number>,
            candidates: (frame.candidates ?? {}) as Record<string, number>,
          };
          setMessages((prev) =>
            prev.map((message) =>
              message.id === frame.turn_id ? { ...message, sources: hits, retrieval: info } : message,
            ),
          );
          break;
        }

        case "delta": {
          const turnId = String(frame.turn_id ?? "");
          if (!turnId) break;
          if (!firstDeltaRef.current.has(turnId)) {
            firstDeltaRef.current.add(turnId);
            const started = startedAtRef.current.get(turnId) ?? Date.now();
            const ttft = Math.max(0, Date.now() - started);
            clientTtftRef.current.set(turnId, ttft);
            setMessages((prev) =>
              prev.map((message) => (message.id === turnId ? { ...message, clientTtftMs: ttft } : message)),
            );
          }
          statsRef.current.deltaFrames += 1;
          statsRef.current.tokens += Number(frame.coalesced ?? 1) || 1;
          queueDelta(turnId, String(frame.text ?? ""));
          break;
        }

        case "tool_call_start": {
          const view: ToolCallView = {
            callId: String(frame.call_id ?? ""),
            name: String(frame.name ?? "tool"),
            arguments: frame.arguments,
            status: "running",
          };
          setMessages((prev) =>
            prev.map((message) =>
              message.id === frame.turn_id
                ? { ...message, tools: [...(message.tools ?? []), view] }
                : message,
            ),
          );
          break;
        }

        case "tool_call_result": {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === frame.turn_id
                ? {
                    ...message,
                    tools: (message.tools ?? []).map((tool) =>
                      tool.callId === frame.call_id
                        ? {
                            ...tool,
                            status: frame.ok ? "ok" : "error",
                            latencyMs: typeof frame.latency_ms === "number" ? frame.latency_ms : undefined,
                            result: frame.result,
                          }
                        : tool,
                    ),
                  }
                : message,
            ),
          );
          break;
        }

        case "done": {
          flushPending();
          const clientTtft = clientTtftRef.current.get(frame.turn_id);
          setMessages((prev) =>
            prev.map((message) =>
              message.id === frame.turn_id
                ? {
                    ...message,
                    streaming: false,
                    clientTtftMs: message.clientTtftMs ?? clientTtft,
                    serverTtftMs: typeof frame.ttft_ms === "number" ? frame.ttft_ms : null,
                    metrics: {
                      finishReason: String(frame.finish_reason ?? "stop"),
                      ttftMs: typeof frame.ttft_ms === "number" ? frame.ttft_ms : null,
                      totalMs: typeof frame.total_ms === "number" ? frame.total_ms : null,
                      usage: frame.usage,
                      model: String(frame.model ?? ""),
                      toolIterations: Number(frame.tool_iterations ?? 0),
                    },
                  }
                : message,
            ),
          );
          statsRef.current.turns += 1;
          statsRef.current.lastTtftMs = typeof frame.ttft_ms === "number" ? frame.ttft_ms : (clientTtft ?? null);
          statsRef.current.lastTotalMs = typeof frame.total_ms === "number" ? frame.total_ms : null;
          if (activeTurnRef.current === frame.turn_id) activeTurnRef.current = null;
          publishStats();
          break;
        }

        case "aborted": {
          flushPending();
          setMessages((prev) =>
            prev.map((message) =>
              message.id === frame.turn_id ? { ...message, streaming: false, aborted: true } : message,
            ),
          );
          if (activeTurnRef.current === frame.turn_id) activeTurnRef.current = null;
          publishStats();
          break;
        }

        case "error": {
          if (frame.turn_id) {
            flushPending();
            setMessages((prev) =>
              prev.map((message) =>
                message.id === frame.turn_id
                  ? {
                      ...message,
                      streaming: false,
                      error: {
                        code: String(frame.code ?? "error"),
                        message: String(frame.message ?? "request failed"),
                        retryable: Boolean(frame.retryable),
                      },
                    }
                  : message,
              ),
            );
            if (activeTurnRef.current === frame.turn_id) activeTurnRef.current = null;
          } else {
            setLastError({
              code: String(frame.code ?? "error"),
              message: String(frame.message ?? "connection error"),
              retryable: Boolean(frame.retryable),
            });
          }
          publishStats();
          break;
        }

        case "ping": {
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "pong", ts: frame.ts ?? null }));
          }
          break;
        }

        case "pong": {
          if (typeof frame.ts === "number") {
            statsRef.current.rttMs = Math.max(0, Date.now() - frame.ts);
            publishStats();
          }
          break;
        }

        case "audio_ack": {
          setAudio((prev) => ({
            ...prev,
            level: Number(frame.rms ?? 0),
            bufferedMs: Number(frame.buffered_ms ?? 0),
            transcribing: Boolean(frame.transcribing),
          }));
          break;
        }

        case "transcript": {
          const text = String(frame.text ?? "").trim();
          if (text) {
            setAudio((prev) => ({ ...prev, transcripts: [...prev.transcripts.slice(-5), text] }));
          }
          break;
        }

        default:
          break;
      }
    },
    [flushPending, publishStats, queueDelta, startPing],
  );

  // ── connection lifecycle ────────────────────────────────────────────────
  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | null = null;

    const scheduleReconnect = () => {
      if (disposed) return;
      const attempt = attemptRef.current;
      attemptRef.current = Math.min(attempt + 1, RECONNECT_MAX_ATTEMPT);
      const ceiling = Math.min(RECONNECT_CAP_MS, RECONNECT_BASE_MS * 2 ** attempt);
      const delay = Math.max(RECONNECT_BASE_MS, Math.round(Math.random() * ceiling));
      setStatus("reconnecting");
      setRetryInMs(delay);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        setRetryInMs(0);
        connect();
      }, delay);
    };

    const connect = () => {
      if (disposed) return;
      const current = wsRef.current;
      if (current && (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      setStatus(attemptRef.current === 0 ? "connecting" : "reconnecting");

      let socket: WebSocket;
      try {
        socket = new WebSocket(resolveSocketUrl());
      } catch {
        scheduleReconnect();
        return;
      }
      wsRef.current = socket;

      socket.onopen = () => {
        socket.send(
          JSON.stringify({
            type: "hello",
            session_id: sessionIdRef.current,
            protocol_version: PROTOCOL_VERSION,
            client: {
              name: "cuepilot-desktop",
              platform: window.cue?.platform ?? "web",
              userAgent: navigator.userAgent.slice(0, 120),
            },
          }),
        );
      };

      socket.onmessage = (event) => {
        if (typeof event.data === "string") handleFrame(event.data);
      };

      socket.onerror = () => {
        /* the close event carries the actionable state */
      };

      socket.onclose = (event) => {
        if (wsRef.current === socket) wsRef.current = null;
        stopPing();
        if (disposed) return;

        setLastClose({ code: event.code, reason: event.reason || "socket closed" });

        if (intentionalCloseRef.current) {
          // Explicit "new session": reconnect immediately instead of backing off.
          intentionalCloseRef.current = false;
          attemptRef.current = 0;
          connect();
          return;
        }
        scheduleReconnect();
      };
    };

    connectRef.current = connect;
    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      stopPing();
      const socket = wsRef.current;
      wsRef.current = null;
      socket?.close(1000, "unmount");
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [handleFrame, stopPing]);

  // ── outbound actions ────────────────────────────────────────────────────
  const sendPrompt = useCallback((text: string, options?: Partial<PromptOptions>): boolean => {
    const socket = wsRef.current;
    const prompt = text.trim();
    if (!socket || socket.readyState !== WebSocket.OPEN || !prompt) return false;

    const turnId = newId();
    const now = Date.now();
    startedAtRef.current.set(turnId, now);
    activeTurnRef.current = turnId;

    setMessages((prev) => [
      ...prev,
      { id: `${turnId}:user`, role: "user", text: prompt, createdAt: now },
      { id: turnId, role: "assistant", text: "", createdAt: now, streaming: true, tools: [] },
    ]);

    socket.send(
      JSON.stringify({
        type: "prompt",
        id: turnId,
        text: prompt,
        options: {
          mode: options?.mode ?? "hybrid",
          tools: options?.tools ?? true,
          ...(options?.top_k ? { top_k: options.top_k } : {}),
          ...(typeof options?.temperature === "number" ? { temperature: options.temperature } : {}),
          ...(options?.max_tokens ? { max_tokens: options.max_tokens } : {}),
        },
      }),
    );
    return true;
  }, []);

  const abort = useCallback(() => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const turnId = activeTurnRef.current;
    socket.send(JSON.stringify(turnId ? { type: "abort", id: turnId } : { type: "abort" }));
  }, []);

  const resetSession = useCallback(() => {
    sessionIdRef.current = null;
    persistSession(null);
    pendingRef.current.clear();
    startedAtRef.current.clear();
    clientTtftRef.current.clear();
    firstDeltaRef.current.clear();
    activeTurnRef.current = null;
    statsRef.current = { ...INITIAL_STATS };
    setStats({ ...INITIAL_STATS });
    setMessages([]);
    setSessionId(null);
    setKbVersion(null);
    setLastError(null);
    setAudio({ ...INITIAL_AUDIO });

    const socket = wsRef.current;
    if (socket) {
      intentionalCloseRef.current = true;
      setStatus("connecting");
      socket.close(1000, "new session");
    } else {
      attemptRef.current = 0;
      connectRef.current?.();
    }
  }, []);

  // ── audio ───────────────────────────────────────────────────────────────
  const sendAudioFrame = useCallback((samples: Int16Array, sampleRate: number) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const seq = audioSeqRef.current++;
    socket.send(
      JSON.stringify({
        type: "audio",
        seq,
        pcm: int16ToBase64(samples),
        sample_rate: Math.round(sampleRate),
      }),
    );
  }, []);

  const detachCapture = useCallback(() => {
    const handles = captureRef.current;
    captureRef.current = null;
    if (handles) {
      handles.node.port.onmessage = null;
      handles.source.disconnect();
      handles.node.disconnect();
      handles.sink.disconnect();
      handles.stream.getTracks().forEach((track) => track.stop());
      void handles.ctx.close().catch(() => undefined);
    }
    setAudio((prev) => ({ ...prev, active: false, source: null, level: 0 }));
  }, []);

  // One graph for both sources: the only difference is which MediaStream feeds it.
  const attachStream = useCallback(
    async (stream: MediaStream, kind: AudioSource) => {
      const AudioCtor = window.AudioContext;
      let ctx: AudioContext;
      try {
        ctx = new AudioCtor({ sampleRate: 16_000 });
      } catch {
        ctx = new AudioCtor();
      }

      const moduleUrl = URL.createObjectURL(new Blob([PCM_WORKLET], { type: "application/javascript" }));
      try {
        await ctx.audioWorklet.addModule(moduleUrl);
      } finally {
        URL.revokeObjectURL(moduleUrl);
      }

      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "pcm-tap");
      const sink = ctx.createGain();
      sink.gain.value = 0; // keep the graph alive without echoing to speakers
      source.connect(node);
      node.connect(sink);
      sink.connect(ctx.destination);

      const chunkSamples = Math.max(256, Math.round(ctx.sampleRate * (AUDIO_CHUNK_MS / 1000)));
      let pending: number[] = [];

      node.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const frame = event.data;
        for (let i = 0; i < frame.length; i += 1) {
          const clamped = Math.max(-1, Math.min(1, frame[i]));
          pending.push(clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff);
        }
        while (pending.length >= chunkSamples) {
          const slice = Int16Array.from(pending.slice(0, chunkSamples));
          pending = pending.slice(chunkSamples);
          sendAudioFrame(slice, ctx.sampleRate);
        }
      };

      captureRef.current = { kind, ctx, stream, source, node, sink };
      setAudio((prev) => ({ ...prev, active: true, source: kind, error: null }));
    },
    [sendAudioFrame],
  );

  const startMic = useCallback(async () => {
    detachCapture();
    if (!navigator.mediaDevices?.getUserMedia) {
      setAudio((prev) => ({ ...prev, error: "microphone is unavailable in this environment" }));
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      await attachStream(stream, "mic");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAudio((prev) => ({ ...prev, active: false, source: null, error: `mic: ${message}` }));
    }
  }, [attachStream, detachCapture]);

  const startSystemAudio = useCallback(async () => {
    detachCapture();

    // The main process owns the capture source; ask it whether loopback is
    // actually available before prompting the user for screen access.
    const bridge = window.cue;
    if (bridge?.getAudioCapabilities) {
      try {
        const caps = await bridge.getAudioCapabilities();
        if (!caps.systemAudio) {
          setAudio((prev) => ({
            ...prev,
            active: false,
            source: null,
            error: caps.reason ?? "system audio capture is unavailable",
          }));
          return;
        }
        if (caps.screenPermission === "denied" || caps.screenPermission === "restricted") {
          setAudio((prev) => ({
            ...prev,
            active: false,
            source: null,
            error: caps.reason ?? "screen recording permission is denied",
          }));
          return;
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setAudio((prev) => ({ ...prev, active: false, source: null, error: `system: ${message}` }));
        return;
      }
    }

    if (!navigator.mediaDevices?.getDisplayMedia) {
      setAudio((prev) => ({ ...prev, error: "system audio is unavailable in this environment" }));
      return;
    }
    try {
      // getDisplayMedia requires a video request even though we only keep audio.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
      });
      stream.getVideoTracks().forEach((track) => {
        track.stop();
        stream.removeTrack(track);
      });
      if (stream.getAudioTracks().length === 0) {
        stream.getTracks().forEach((track) => track.stop());
        setAudio((prev) => ({
          ...prev,
          active: false,
          source: null,
          error:
            "no system-audio track was returned. Grant Screen Recording and System Audio " +
            "Recording permission, then retry.",
        }));
        return;
      }
      await attachStream(stream, "system");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAudio((prev) => ({ ...prev, active: false, source: null, error: `system: ${message}` }));
    }
  }, [attachStream, detachCapture]);

  const toggleMic = useCallback(() => {
    if (captureRef.current?.kind === "mic") detachCapture();
    else void startMic();
  }, [detachCapture, startMic]);

  const toggleSystemAudio = useCallback(() => {
    if (captureRef.current?.kind === "system") detachCapture();
    else void startSystemAudio();
  }, [detachCapture, startSystemAudio]);

  useEffect(() => () => detachCapture(), [detachCapture]);

  const activeTurnId = messages.find((message) => message.streaming)?.id ?? null;

  return {
    status,
    retryInMs,
    lastError,
    lastClose,
    sessionId,
    model,
    kbVersion,
    messages,
    stats,
    audio,
    activeTurnId,
    sendPrompt,
    abort,
    resetSession,
    toggleMic,
    toggleSystemAudio,
    clearError: () => setLastError(null),
  };
}

export type CueStream = ReturnType<typeof useCueStream>;

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useCueStream,
  type ChatMessage,
  type ConnectionStatus,
  type RetrievalMode,
  type SourceHit,
  type StreamStats,
} from "./useCueStream";

// ── formatting ────────────────────────────────────────────────────────────
function formatMs(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

function formatCount(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString();
}

function clockTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const SUGGESTIONS = [
  {
    title: "Trace a slow request",
    body: "How is TTFT measured, and which span tells me the gateway overhead?",
  },
  {
    title: "Explain reconnect",
    body: "Why does the client use full jitter on reconnect, and what happens to a partial answer?",
  },
  {
    title: "Compare retrieval",
    body: "Compare BM25 and dense retrieval on this corpus. When does RRF help most?",
  },
  {
    title: "Debug an error code",
    body: "CUE-E3010 keeps appearing after an embedding model change. What should I check?",
  },
];

const QUICK_PROMPTS = [
  "How does the delta coalescer shed frame pressure without dropping tokens?",
  "Walk me through the tool-calling loop and its timeout handling.",
  "What does CUE-E2041 mean and how should the client retry?",
  "How are audio frames turned into transcript windows?",
];

const MODES: { value: RetrievalMode; label: string }[] = [
  { value: "hybrid", label: "hybrid" },
  { value: "bm25", label: "bm25" },
  { value: "dense", label: "dense" },
];

// ── small presentational pieces ───────────────────────────────────────────
function StatusPill({ status, retryInMs }: { status: ConnectionStatus; retryInMs: number }) {
  const label =
    status === "open"
      ? "connected"
      : status === "connecting"
        ? "connecting"
        : status === "reconnecting"
          ? `retry in ${Math.max(1, Math.ceil(retryInMs / 1000))}s`
          : "offline";
  return (
    <span className={`status-pill status-${status}`}>
      <span className="status-dot" />
      {label}
    </span>
  );
}

function RichText({ text }: { text: string }) {
  const parts = text.split("```");
  return (
    <>
      {parts.map((part, index) => {
        if (index % 2 === 1) {
          const newline = part.indexOf("\n");
          const language = newline >= 0 ? part.slice(0, newline).trim() : "";
          const code = newline >= 0 ? part.slice(newline + 1) : part;
          return (
            <pre className="code-block" key={index}>
              {language ? <span className="code-lang">{language}</span> : null}
              <code>{code.replace(/\n$/, "")}</code>
            </pre>
          );
        }
        return (
          <span className="prose" key={index}>
            {part}
          </span>
        );
      })}
    </>
  );
}

function ToolChip({
  name,
  status,
  latencyMs,
  args,
}: {
  name: string;
  status: "running" | "ok" | "error";
  latencyMs?: number;
  args: unknown;
}) {
  const title = useMemo(() => {
    try {
      return JSON.stringify(args, null, 2).slice(0, 400);
    } catch {
      return String(args);
    }
  }, [args]);

  return (
    <span className={`tool-chip tool-${status}`} title={title}>
      <span className="tool-glyph">{status === "running" ? "◌" : status === "ok" ? "✓" : "!"}</span>
      <span className="tool-name">{name}</span>
      {latencyMs !== undefined ? <span className="tool-latency">{formatMs(latencyMs)}</span> : null}
    </span>
  );
}

function SourceList({
  hits,
  retrieval,
}: {
  hits: SourceHit[];
  retrieval?: ChatMessage["retrieval"];
}) {
  if (!hits.length) return null;
  const max = Math.max(...hits.map((hit) => hit.score), 0.000001);
  return (
    <section className="sources">
      <header className="sources-head">
        <span>Retrieved context</span>
        <span className="sources-sub">
          {hits.length} chunks · {retrieval?.mode ?? "hybrid"}
          {retrieval?.cache_hit ? " · cache hit" : ""}
          {retrieval?.embedder ? ` · ${retrieval.embedder}` : ""}
        </span>
      </header>
      <div className="source-grid">
        {hits.map((hit, index) => {
          const ranks = Object.entries(hit.explain?.ranks ?? {});
          return (
            <article className="source-card" key={hit.id}>
              <div className="source-top">
                <span className="source-rank">[{index + 1}]</span>
                <span className="source-score">{hit.score.toFixed(4)}</span>
              </div>
              <div className="source-title">{hit.title}</div>
              <div className="source-id">{hit.doc_id}</div>
              <div className="source-ranks">
                {ranks.length ? (
                  ranks.map(([retriever, rank]) => (
                    <span className="rank-badge" key={retriever}>
                      {retriever} #{rank}
                    </span>
                  ))
                ) : (
                  <span className="rank-badge muted">no provenance</span>
                )}
              </div>
              <div className="source-bar">
                <span style={{ width: `${Math.max(4, (hit.score / max) * 100)}%` }} />
              </div>
            </article>
          );
        })}
      </div>
      {retrieval?.degraded?.length ? (
        <div className="degraded-note">
          degraded path: {retrieval.degraded.join(", ")} — serving partial recall
        </div>
      ) : null}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </span>
  );
}

function SkeletonLines() {
  return (
    <div className="skeleton" aria-label="generating">
      <span />
      <span />
      <span />
    </div>
  );
}

function AssistantTurn({
  message,
  onRetry,
}: {
  message: ChatMessage;
  onRetry: (message: ChatMessage) => void;
}) {
  const usage = message.metrics?.usage;
  const hasText = message.text.length > 0;

  return (
    <article className="turn turn-assistant">
      <header className="turn-head">
        <span className="avatar">◆</span>
        <span className="turn-role">CuePilot</span>
        {message.streaming ? <span className="live-dot" /> : null}
        <span className="turn-time">{clockTime(message.createdAt)}</span>
      </header>

      {message.tools?.length ? (
        <div className="tool-row">
          {message.tools.map((tool) => (
            <ToolChip
              key={tool.callId}
              name={tool.name}
              status={tool.status}
              latencyMs={tool.latencyMs}
              args={tool.arguments}
            />
          ))}
        </div>
      ) : null}

      <div className="answer">
        {hasText ? <RichText text={message.text} /> : message.streaming ? <SkeletonLines /> : null}
        {message.streaming && hasText ? <span className="caret" /> : null}
        {!hasText && !message.streaming && !message.error && !message.aborted ? (
          <span className="muted">No answer text.</span>
        ) : null}
      </div>

      {message.error ? (
        <div className="inline-error">
          <span className="inline-error-code">{message.error.code}</span>
          <span>{message.error.message}</span>
          {message.error.retryable ? (
            <button type="button" className="link-btn" onClick={() => onRetry(message)}>
              Retry
            </button>
          ) : null}
        </div>
      ) : null}

      {message.aborted ? <div className="inline-note">Generation stopped.</div> : null}
      {message.interrupted && !message.streaming ? (
        <div className="inline-note">
          Partial answer restored after an interruption
          {message.checkpointedAt ? ` · checkpointed ${clockTime(message.checkpointedAt)}` : ""}.
        </div>
      ) : null}

      <SourceList hits={message.sources ?? []} retrieval={message.retrieval} />

      {message.metrics || message.clientTtftMs !== undefined ? (
        <footer className="turn-meta">
          <Metric label="ttft (client)" value={formatMs(message.clientTtftMs)} />
          <Metric label="ttft (server)" value={formatMs(message.serverTtftMs)} />
          <Metric label="total" value={formatMs(message.metrics?.totalMs)} />
          <Metric label="tokens" value={formatCount(usage?.total_tokens)} />
          <Metric label="tools" value={formatCount(message.metrics?.toolIterations)} />
          <Metric label="finish" value={message.metrics?.finishReason ?? "—"} />
        </footer>
      ) : null}
    </article>
  );
}

function EmptyState({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="empty">
      <div className="empty-kicker">hybrid retrieval · streamed tools · measured</div>
      <h1>
        Ask the <span>codebase</span>.
      </h1>
      <p>
        Every answer streams token-by-token over one full-duplex socket, grounded in BM25 + pgvector
        results fused with reciprocal rank fusion. Tool calls, sources and latency stay visible.
      </p>
      <div className="suggestion-grid">
        {SUGGESTIONS.map((suggestion) => (
          <button
            type="button"
            className="suggestion"
            key={suggestion.title}
            disabled={disabled}
            onClick={() => onPick(suggestion.body)}
          >
            <span className="suggestion-title">{suggestion.title}</span>
            <span className="suggestion-body">{suggestion.body}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function StatsPanel({ stats, model }: { stats: StreamStats; model: string }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <span>Live metrics</span>
        <span className="panel-sub">this connection</span>
      </header>
      <div className="stat-grid">
        <div className="stat">
          <span className="stat-label">last ttft</span>
          <span className="stat-value">{formatMs(stats.lastTtftMs)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">last turn</span>
          <span className="stat-value">{formatMs(stats.lastTotalMs)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">rtt</span>
          <span className="stat-value">{formatMs(stats.rttMs)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">turns</span>
          <span className="stat-value">{formatCount(stats.turns)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">frames</span>
          <span className="stat-value">{formatCount(stats.frames)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">delta frames</span>
          <span className="stat-value">{formatCount(stats.deltaFrames)}</span>
        </div>
      </div>
      <div className="stat-foot">
        <span>{formatCount(stats.tokens)} tokens coalesced</span>
        <span className="mono">{model}</span>
      </div>
    </section>
  );
}

function ContextPanel({ message }: { message?: ChatMessage }) {
  const retrieval = message?.retrieval;
  return (
    <section className="panel">
      <header className="panel-head">
        <span>Context</span>
        <span className="panel-sub">{retrieval ? retrieval.mode : "idle"}</span>
      </header>
      {retrieval ? (
        <>
          <div className="context-query">“{retrieval.query}”</div>
          <div className="context-rows">
            <div className="context-row">
              <span>kb version</span>
              <span className="mono">v{retrieval.kb_version}</span>
            </div>
            <div className="context-row">
              <span>cache</span>
              <span className={retrieval.cache_hit ? "good" : "muted"}>
                {retrieval.cache_hit ? "hit" : "miss"}
              </span>
            </div>
            <div className="context-row">
              <span>candidates</span>
              <span className="mono">
                {Object.entries(retrieval.candidates)
                  .map(([key, value]) => `${key} ${value}`)
                  .join(" · ") || "—"}
              </span>
            </div>
            <div className="context-row">
              <span>retrieval</span>
              <span className="mono">
                {formatMs(
                  retrieval.timings_ms.total ??
                    Math.max(0, ...Object.values(retrieval.timings_ms).filter(Number.isFinite)),
                )}
              </span>
            </div>
          </div>
          <ul className="context-hits">
            {(message?.sources ?? []).slice(0, 6).map((hit, index) => (
              <li key={hit.id}>
                <span className="context-hit-rank">{index + 1}</span>
                <span className="context-hit-body">
                  <span className="context-hit-title">{hit.title}</span>
                  <span className="context-hit-id">{hit.doc_id}</span>
                </span>
                <span className="context-hit-score">{hit.score.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="panel-empty">Retrieved chunks appear here as a turn starts.</p>
      )}
    </section>
  );
}

function SessionPanel({
  sessionId,
  model,
  kbVersion,
  onCopy,
}: {
  sessionId: string | null;
  model: string;
  kbVersion: number | null;
  onCopy: (value: string) => void;
}) {
  return (
    <section className="panel">
      <header className="panel-head">
        <span>Session</span>
        <span className="panel-sub">resumable</span>
      </header>
      <div className="context-rows">
        <div className="context-row">
          <span>id</span>
          <button
            type="button"
            className="mono id-copy"
            title={sessionId ?? ""}
            onClick={() => sessionId && onCopy(sessionId)}
          >
            {sessionId ? `${sessionId.slice(0, 10)}…` : "new"}
          </button>
        </div>
        <div className="context-row">
          <span>model</span>
          <span className="mono">{model}</span>
        </div>
        <div className="context-row">
          <span>kb</span>
          <span className="mono">{kbVersion !== null ? `v${kbVersion}` : "—"}</span>
        </div>
        <div className="context-row">
          <span>protocol</span>
          <span className="mono">v1</span>
        </div>
      </div>
    </section>
  );
}

// ── app ───────────────────────────────────────────────────────────────────
export default function App() {
  const {
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
    clearError,
  } = useCueStream();

  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [toolsEnabled, setToolsEnabled] = useState(true);
  const [toast, setToast] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const isMac = window.cue?.platform === "darwin";

  const latestAssistant = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant"),
    [messages],
  );

  const canSend = draft.trim().length > 0 && status === "open";
  const isStreaming = Boolean(activeTurnId);

  // keep the transcript pinned to the newest token unless the user scrolled up
  useEffect(() => {
    const node = transcriptRef.current;
    if (node && stickToBottomRef.current) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages]);

  // auto-grow composer
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(220, node.scrollHeight)}px`;
  }, [draft]);

  // Electron main-process events: hotkey context and non-fatal warnings
  useEffect(() => {
    const bridge = window.cue;
    if (!bridge) return;
    const offHotkey = bridge.onHotkey((payload) => {
      const selection = payload.selection?.trim();
      setDraft((current) => (selection ? `${selection}\n\n${current}` : current));
      setToast(
        selection
          ? `Captured ${selection.length} chars from ${payload.windowTitle || "the active window"}`
          : "Hotkey fired — no text selected",
      );
      window.setTimeout(() => textareaRef.current?.focus(), 0);
    });
    const offWarning = bridge.onWarning((payload) => setToast(payload.message));
    return () => {
      offHotkey();
      offWarning();
    };
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const submit = () => {
    if (!canSend) return;
    if (sendPrompt(draft, { mode, tools: toolsEnabled })) setDraft("");
  };

  const retry = (message: ChatMessage) => {
    const index = messages.findIndex((item) => item.id === message.id);
    for (let i = index - 1; i >= 0; i -= 1) {
      if (messages[i].role === "user") {
        sendPrompt(messages[i].text, { mode, tools: toolsEnabled });
        return;
      }
    }
  };

  const pickSuggestion = (text: string) => {
    setDraft(text);
    textareaRef.current?.focus();
  };

  const copy = (value: string) => {
    void navigator.clipboard
      .writeText(value)
      .then(() => setToast("Session id copied"))
      .catch(() => setToast("Clipboard unavailable"));
  };

  return (
    <div className="app">
      <header className={`titlebar${isMac ? " mac" : ""}`}>
        <div className="brand">
          <span className="brand-mark">◆</span>
          <span className="brand-name">
            cue<span>Pilot</span>
          </span>
          <span className="brand-tag">realtime copilot</span>
        </div>
        <div className="titlebar-right">
          <StatusPill status={status} retryInMs={retryInMs} />
          <button type="button" className="ghost-btn" onClick={resetSession}>
            New session
          </button>
        </div>
      </header>

      {lastError ? (
        <div className="banner banner-error">
          <span>
            <strong>{lastError.code}</strong> {lastError.message}
          </span>
          <button type="button" className="link-btn" onClick={clearError}>
            Dismiss
          </button>
        </div>
      ) : null}

      {status === "reconnecting" ? (
        <div className="banner banner-warn">
          <span>
            Reconnecting in {Math.max(1, Math.ceil(retryInMs / 1000))}s · last close{" "}
            <span className="mono">{lastClose?.code ?? "—"}</span> {lastClose?.reason ?? ""}
          </span>
          <span className="banner-hint">session will resume automatically</span>
        </div>
      ) : null}

      <div className="shell">
        <aside className="rail">
          <div className="rail-block">
            <div className="rail-label">Try asking</div>
            <div className="quick-list">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  type="button"
                  className="quick"
                  key={prompt}
                  onClick={() => pickSuggestion(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <div className="rail-block">
            <div className="rail-label">Pipeline</div>
            <ul className="pipeline">
              <li>
                <span className="pipe-step">1</span>
                <span>
                  <strong>Hybrid retrieval</strong>
                  <em>BM25 + pgvector, RRF k=60</em>
                </span>
              </li>
              <li>
                <span className="pipe-step">2</span>
                <span>
                  <strong>Tool loop</strong>
                  <em>kb_search · kb_open · kb_stats</em>
                </span>
              </li>
              <li>
                <span className="pipe-step">3</span>
                <span>
                  <strong>Streamed answer</strong>
                  <em>coalesced at 16 ms</em>
                </span>
              </li>
            </ul>
          </div>

          <div className="rail-foot">
            <kbd>⌘</kbd>
            <kbd>⇧</kbd>
            <kbd>Space</kbd>
            <span>capture selection</span>
          </div>
        </aside>

        <main className="stage">
          <div
            className="transcript"
            ref={transcriptRef}
            onScroll={(event) => {
              const node = event.currentTarget;
              stickToBottomRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
            }}
          >
            {messages.length === 0 ? (
              <EmptyState onPick={pickSuggestion} disabled={status !== "open"} />
            ) : (
              <div className="turns">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <div className="turn turn-user" key={message.id}>
                      <div className="user-bubble">{message.text}</div>
                      <span className="turn-time user-time">{clockTime(message.createdAt)}</span>
                    </div>
                  ) : (
                    <AssistantTurn key={message.id} message={message} onRetry={retry} />
                  ),
                )}
              </div>
            )}
          </div>

          {audio.transcripts.length ? (
            <div className="audio-strip">
              <span className="audio-label">audio</span>
              <span className="audio-text">{audio.transcripts[audio.transcripts.length - 1]}</span>
            </div>
          ) : null}

          <div className="composer-wrap">
            <div className={`composer${isStreaming ? " streaming" : ""}`}>
              <textarea
                ref={textareaRef}
                value={draft}
                rows={1}
                placeholder={
                  status === "open"
                    ? "Ask about the codebase, an error code, or a design decision…"
                    : "Waiting for the gateway…"
                }
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submit();
                  }
                }}
              />
              <div className="composer-bar">
                <div className="composer-left">
                  <label className="select-wrap" title="Retrieval mode">
                    <select
                      value={mode}
                      onChange={(event) => setMode(event.target.value as RetrievalMode)}
                    >
                      {MODES.map((option) => (
                        <option value={option.value} key={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className={`chip-toggle${toolsEnabled ? " on" : ""}`}
                    onClick={() => setToolsEnabled((value) => !value)}
                    aria-pressed={toolsEnabled}
                  >
                    tools
                  </button>
                  <button
                    type="button"
                    className={`chip-toggle mic${audio.source === "mic" ? " on" : ""}`}
                    onClick={toggleMic}
                    title={audio.error ?? "Stream raw PCM from the microphone"}
                    aria-pressed={audio.source === "mic"}
                  >
                    <span className="mic-dot" style={{ opacity: 0.35 + Math.min(1, audio.level * 12) }} />
                    mic
                  </button>
                  <button
                    type="button"
                    className={`chip-toggle mic${audio.source === "system" ? " on" : ""}`}
                    onClick={toggleSystemAudio}
                    title="Stream system audio (what the call or video is playing) as raw PCM"
                    aria-pressed={audio.source === "system"}
                  >
                    <span className="mic-dot" style={{ opacity: 0.35 + Math.min(1, audio.level * 12) }} />
                    system
                  </button>
                  {audio.active ? (
                    <span className="level-meter" aria-hidden>
                      <span style={{ width: `${Math.min(100, audio.level * 260)}%` }} />
                    </span>
                  ) : null}
                </div>
                <div className="composer-right">
                  <span className="hint">Enter to send · Shift+Enter newline</span>
                  {isStreaming ? (
                    <button type="button" className="send stop" onClick={abort}>
                      Stop
                    </button>
                  ) : (
                    <button type="button" className="send" onClick={submit} disabled={!canSend}>
                      Send
                    </button>
                  )}
                </div>
              </div>
            </div>
            {audio.error ? <div className="composer-note">{audio.error}</div> : null}
          </div>
        </main>

        <aside className="inspector">
          <StatsPanel stats={stats} model={model} />
          <ContextPanel message={latestAssistant} />
          <SessionPanel sessionId={sessionId} model={model} kbVersion={kbVersion} onCopy={copy} />
        </aside>
      </div>

      {toast ? (
        <button type="button" className="toast" onClick={() => setToast(null)}>
          {toast}
        </button>
      ) : null}
    </div>
  );
}

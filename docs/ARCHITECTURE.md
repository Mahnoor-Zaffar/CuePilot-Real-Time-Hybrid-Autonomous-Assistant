# Architecture

This document explains how cuePilot is put together and why. For the wire format see
[PROTOCOL.md](PROTOCOL.md); for what happens when things break see
[FAILURE_MODES.md](FAILURE_MODES.md); for numbers see [METRICS.md](METRICS.md).

## Goals

1. **One ordered, full-duplex channel.** Text deltas, tool events, and terminal frames
   share a single WebSocket so ordering is a property of the transport, not of a merge
   algorithm.
2. **Real backpressure.** A slow client must degrade frame rate, never corrupt or drop
   answer text.
3. **Retrieval that can explain itself.** Every hit carries per-retriever rank,
   contribution, and raw score.
4. **Runs offline.** No API key, no Docker, no model download: the full pipeline still
   works, which makes tests and benchmarks reproducible.
5. **Honest degradation.** Fallbacks are reported in `/api/health`, `/api/metrics`, and
   in the UI.

Non-goals: multi-tenant authz, model fine-tuning.

## Component map

```
┌──────────────────────────── Electron (renderer sandboxed) ────────────────────────────┐
│  React UI (App.tsx)                                                                   │
│    └── useCueStream.ts ── rAF delta batching ── WebSocket ── AudioWorklet → PCM       │
│  ▲ contextBridge (preload.ts): hotkeys, clipboard, window context — no Node globals   │
└───┼───────────────────────────────────────────────────────────────────────────────────┘
    │  ws://127.0.0.1:8000/ws/stream        JSON frames, ordered, full duplex
┌───▼───────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI gateway (app/main.py, app/runtime.py)                                        │
│                                                                                       │
│  Connection (ws/connection.py)                                                        │
│    ├── handshake + auth + heartbeat + close codes                                     │
│    ├── ClientLimiter (ws/limits.py): shared attempt window + per-replica cap          │
│    ├── OutboundStream (ws/outbound.py): per-turn buffers, 16 ms flush, bounded queue  │
│    └── SessionManager (ws/session.py): hot dict + Redis mirror + Postgres journal     │
│                                                                                       │
│  Agent (llm/agent.py)                                                                 │
│    retrieval → prompt assembly → provider stream → tool loop → done                   │
│         │                                    │                                        │
│  HybridRetriever (retrieval/orchestrator.py)  Provider (llm/registry.py)              │
│    BM25 ∥ dense → RRF                         OpenRouter SSE │ Mock │ FallbackChain    │
│         │                                    │                                        │
│  Infra: Redis cache + pub/sub · pgvector / memory store · hash / HTTP / local embedder  │
│  Schema: numbered SQL migrations, advisory-locked, applied at boot                     │
│  Telemetry: OTel spans + bounded in-memory trace ring                                 │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

The dependency graph is built once in `Runtime.create()` during the FastAPI lifespan and
stored on `app.state.runtime`. Routes and the WebSocket handler take it as a dependency;
nothing constructs its own clients per request.

## Turn lifecycle

A prompt produces this sequence. Numbers are the order frames are written to the socket,
not necessarily the order work completes.

1. **Client** assigns a turn id, appends an optimistic user message and an empty assistant
   message, records `startedAt`, and sends `{type:"prompt", id, text, options}`.
2. **Connection** acknowledges with `{type:"ack", turn_id}`. If a turn is already active it
   is aborted first (`superseded`); one active turn per session.
3. A per-turn `asyncio.Task` starts `Agent.run()`.
4. **Agent** warms the provider pool (no-op for OpenRouter, whose keep-alive client exists
   already) while concurrently running retrieval. Retrieval failure is logged and the turn
   continues without context.
5. **Retriever** emits `{type:"retrieval", hits, timings_ms, candidates, degraded, ...}`.
   Hits include `explain.ranks/contributions/raw_scores`; content is omitted unless the
   caller asks for it.
6. **Agent** builds the system prompt with the retrieved context inside `<context>` and
   replays the last `turn_history_limit` (24) messages.
7. **Provider** streams. On the first delta the agent records TTFT
   (`prompt decode → first delta yielded to the socket layer`) and the outbound stream
   flushes immediately, bypassing the 16 ms window.
8. If the provider emits tool calls, the agent emits `tool_call_start`, dispatches all
   calls concurrently (bounded by `tool_timeout_s`), emits `tool_call_result` per call,
   appends tool messages, and loops (max `max_tool_iterations`, 4).
9. **Agent** emits `done` with `ttft_ms`, `total_ms`, `finish_reason`, usage, model, and
   tool iterations. The outbound stream flushes buffered text before any control frame.
10. **Session** mirrors history to Redis (last 64 entries; replay sends the last 20) and
    enqueues the finished turn to the Postgres journal. Neither write is on the token path.

Cancellation can happen at any point after step 3: `abort` cancels the task, the provider
generator unwinds and closes the upstream HTTP response, buffered text is flushed, and
`aborted` is sent. See [FAILURE_MODES.md](FAILURE_MODES.md#cancellation).

## Retrieval

### Tokenization

`retrieval/tokenize.py` lowercases, splits snake_case and camelCase while keeping the
compound, normalizes error codes (`CUE-E1042`, `cue_e1042`, `e1042` match), drops
stopwords and one-character tokens, and produces bounded character trigrams for the hash
embedder.

### BM25

A classic BM25 index (`k1=1.5`, `b=0.75`) built in a thread from the vector store's chunks.
The store is the source of truth; BM25 is a derived index rebuilt on `refresh()`. Query
search runs in `asyncio.to_thread` under `retrieval_timeout_s` (6 s) so a large corpus cannot
stall the event loop.

### Dense

- `HttpEmbedder`: any OpenAI-compatible `/embeddings` endpoint, batched 64 at a time with
  retries and a dimension check.
- `LocalEmbedder`: `model2vec` static sentence embeddings — distilled token vectors, no
  transformer forward pass, no PyTorch. Selected with
  `CUEPILOT_EMBEDDING_PROVIDER=local`; the model's own width becomes the schema width.
  This is a real semantic encoder, and the same class the benchmark uses, so the offline
  path and the measured path are the same code.
- `HashEmbedder`: deterministic signed feature hashing over word unigrams (sublinear tf),
  bigrams, and character trigrams. It is a **stand-in**, not a semantic model; it exists so
  the dense path, RRF, and benchmarks run offline with no download.

`PgVectorStore` uses cosine distance (`<=>`) with an HNSW index (`m=16`,
`ef_construction=64`). `MemoryVectorStore` does exact cosine over a NumPy matrix; it is
correct and fast below ~100k chunks, and it is the fallback when Postgres is unreachable.

### Schema and migrations

`app/migrations/NNN_name.sql` is the source of truth. `infra/migrations.py` applies files in
order under `pg_advisory_lock`, recording `version`, `checksum`, and the parameters each
file was rendered with in `schema_migrations`. Two consequences worth knowing:

- **Editing an applied file fails startup.** Add `002_...sql` instead.
- **Changing a parameter is a rebuild.** `{{embedding_dim}}` lives in `VECTOR(n)`, which
  pgvector cannot cast between widths, so the runner applies the matching `.down.sql`
  (drop the derived table, bump `kb_meta.version`), re-runs the file, and lets boot
  re-ingest. No down file means a hard error, never a silent no-op.

The gateway migrates before serving; `python scripts/migrate.py` does the same as its own
step. A migration failure is fatal even when `allow_degraded_infra=true`: falling back to
memory would hide a broken schema.

### Fusion

```
score(d) = Σ_r  weight_r / (k + rank_r(d)),   k = 60
```

RRF is chosen over a cross-encoder because it needs no training, no second forward pass,
and is invariant to the score scales of BM25 and cosine. Candidate pool is 50 per retriever;
the fused list is truncated to `retrieval_top_k` (8), and a document may contribute at most
`retrieval_max_chunks_per_doc` chunks (default 1) so one long document cannot fill the
window. Each `FusedHit` keeps the ranks and contributions that produced it, which is what
the UI's source cards display.

A single-retriever mode still runs through RRF so ordering and provenance shapes are
identical across modes.

### Cache and invalidation

Cache key: `cuepilot:ret:{kb_version}:{sha1(query|mode|top_k|candidate_k)}`, TTL 120 s.
Because `kb_version` is in the key, an ingest self-invalidates every prior result. The
ingest endpoint also publishes `kb.invalidate` on the bus so other replicas rebuild their
BM25 index; the origin instance skips its own event.

### Degradation

If one retriever raises (timeout, embedding 5xx), the orchestrator records it in
`degraded`, logs the failure metric, and fuses whatever is left. If both fail, it returns
an empty result with `degraded: ["bm25","dense"]` and the agent answers without context
rather than failing the turn.

## Streaming and backpressure

There are three separate mechanisms; conflating them is the usual source of dropped text.

### Server: coalescing outbound writer

`OutboundStream` owns a single writer task, a bounded queue (`ws_send_queue_size`, 64), and
per-turn text buffers.

- `push_delta` appends text to the turn buffer. The **first** delta flushes synchronously so
  TTFT is not delayed by a flush window. Later deltas set an event.
- A flush task wakes every `delta_flush_ms` (16 ms), joins the buffer, and enqueues one
  `delta` frame carrying `coalesced` (token count). This caps frame rate near 60/s
  regardless of token rate.
- If the queue is full, the delta **frame** is dropped and its text is re-inserted at the
  head of the buffer for the next flush. `delta_frames_dropped` increments; text is intact.
  A hard cap (`MAX_BUFFER_CHARS`, 64k) protects memory if the client is gone; only then are
  characters dropped, and `delta_chars_dropped` records it.
- Control frames (`retrieval`, `tool_call_*`, `done`, `error`, `aborted`) use `send()`, which
  waits up to 3 s for queue space instead of dropping. Before any control frame the turn is
  flushed (`flush_turn`) so text cannot be reordered behind `done`.
- The writer task bounds each `send_json` at 5 s; a wedged socket sets `_closed` and the
  connection tears down.

### Client: animation-frame batching

`useCueStream` accumulates delta text per turn in a `Map` and schedules one
`requestAnimationFrame` callback. On the frame it applies all pending text in a single
React state update. Terminal frames (`done`, `error`, `aborted`, `history`) call
`flushPending()` synchronously first, so the last tokens always render before the turn is
marked complete. The first delta is painted immediately (no batching) to keep the
user-visible TTFT honest.

### Ordering invariant

For a given turn the client observes:
`ack → retrieval? → (delta | tool_call_start | tool_call_result)* → done | aborted | error`.
Within text, `seq` is monotonic per turn. The server guarantees the terminal frame is
enqueued after the final text frame; the client guarantees it is applied after pending text.

## Cancellation

- The client sends `{type:"abort", id?}`. With no id, the session's active turn is aborted.
- `Connection._abort_active` cancels the task and waits up to 2 s for it to unwind.
- `_run_turn` catches `CancelledError`, calls `task.uncancel()` before any post-cancel
  `await` (otherwise the awaits would immediately re-raise), flushes text, sends
  `aborted`, and increments `turns_aborted`.
- The provider's `async for` is inside an `async with client.stream(...)`; cancellation
  closes the HTTP response, so the upstream request stops billing tokens.
- If the abort arrives on a different replica, the connection publishes `turn.abort` on the
  bus with the session id and turn id; the owning replica cancels the matching task.

## Sessions and resume

Three tiers, each with one job:

1. **Hot dict.** An in-memory `Session` is the working copy for an open connection.
2. **Redis mirror.** `cuepilot:session:{id}` holds history (capped at 64), scratchpad,
   `kb_version`, turn count, and client metadata, with `session_ttl_s` (3600 s). This is
   the fast path: a resume is one GET. A sweeper evicts hot sessions past TTL with no
   active turn.
3. **Postgres journal.** `sessions` is the header row and `turns` is one row per finished
   turn (prompt, answer, status, finish reason, timings, tool iterations, usage, model).
   This is the record.

Writes are **write-behind**. `save()` writes Redis synchronously, so the fast path stays
coherent, and enqueues the Postgres upsert on a bounded `asyncio.Queue` drained by one
worker.

A turn has two write classes:

- **Checkpoints** (`status = "streaming"`) are enqueued every
  `turn_checkpoint_interval_ms` (250 ms). They are droppable: the next checkpoint carries
  the whole answer so far, so shedding one costs at most one interval of crash protection.
- **Terminal writes** (`done`, `aborted`, `error`) are the transcript. When the queue is full
  or the write fails after its retry, the record is appended to a bounded local NDJSON spool
  (first) and, if that is unavailable or full, to the shared cache as
  `cuepilot:spool:turn:*` with a TTL of `session_journal_spool_cache_ttl_s`. Both are
  replayed in the background (`session_journal_spool_retry_s`, 30 s) and once during
  graceful shutdown. The local spool is capped at `session_journal_spool_max_bytes` (8 MB);
  when both the local spool and the cache fail, `durability_degraded` is set and new prompts
  are refused with retryable `journal_unavailable` until a write or replay succeeds. The
  cache fallback's durability depends on Redis persistence, which is the operator's
  responsibility.

After a successful terminal write, the in-flight checkpoint key (`cuepilot:inflight:{session_id}`)
is deleted by the journal. Until then it carries the newest partial text with a
`checkpointed_at` epoch timestamp, so a crash between the final checkpoint and the terminal
write still leaves a recoverable partial.

The stream never awaits durability: if Postgres stalls, checkpoints drop
(`journal.dropped`) and terminal writes spill, which costs a little latency in the worst
case rather than the turn. `turns.updated_at` is what makes stale recovery possible: a
`streaming` row older than `turn_checkpoint_stale_s` (60 s) is marked
`aborted`/`interrupted` on the read path, so no separate janitor is needed.

Reads go hot dict → Redis → Postgres → new session. A Redis miss is a cache miss, not data
loss: the journal rebuilds the header plus the most recent `turn_history_limit` turns, and
the connection replays history exactly as it would after a Redis hit. Replayed turns that
were recovered from a checkpoint carry `interrupted: true` and a `partial` flag with a
`checkpointed_at` timestamp. The in-flight key is merged last when present, so the newest
partial survives even when the journal checkpoint is one interval behind. With no Postgres
the journal is absent (`journal.backend = "none"`) and the manager is mirror-only, which the
health endpoint reports instead of hiding.

On connect the client sends its stored session id. If found anywhere in that chain, the
server marks the connection `resumed: true` and replays `history` (last 20 messages). If
not, it creates a new session and returns the new id; the client replaces its stored id.
Session ids are opaque `uuid4().hex`.

## Audio path

Both sources share one capture graph; only the `MediaStream` differs.

1. **Microphone.** `getUserMedia` (mono, echo cancellation, noise suppression).
2. **System audio.** The renderer calls `getDisplayMedia({video: true, audio: true})` and
   stops the video track immediately. The main process answers with
   `setDisplayMediaRequestHandler` instead of a picker: it accepts only the top-level
   frame at the app origin and passes `{video: screen, audio: "loopback"}`.
3. Either stream feeds `AudioContext` at 16 kHz when the platform allows it. An inline
   AudioWorklet module (`pcm-tap`) posts Float32 blocks to the main thread.
4. The hook clamps and converts to Int16, accumulates until 100 ms of samples, base64
   encodes, and sends `{type:"audio", seq, pcm, sample_rate}`.
5. The server appends to a bounded `AudioBuffer` (3 s), returns `audio_ack` with buffered
   duration and RMS (used by the level meter), and — when an OpenAI key is configured —
   starts a rolling-window Whisper transcription. Transcripts arrive as `transcript`
   frames. Transcription is best effort and never blocks the receive loop.
6. A zero-gain `GainNode` keeps the graph alive without echoing the mic to speakers.

Loopback is native on Windows and on Electron 39+. This build pins Electron 33, so
`main.ts` appends `MacLoopbackAudioForScreenShare` and `MacSckSystemAudioLoopbackOverride`
to `enable-features` before `app.whenReady()`. Without them macOS returns a track that ends
immediately — a silent failure, which is why the capability IPC reports the permission
state and a reason the renderer can display. Linux is not claimed: it depends on the sound
server (PulseAudio/PipeWire), and the IPC says so instead of pretending.

## Provider layer

`ChatProvider` yields typed `StreamEvent`s: `delta`, `tool_calls`, `usage`, `finish`.

`OpenRouterProvider`:
- OpenAI-compatible SSE (`/chat/completions`, `stream: true`, `include_usage`).
- Reassembles streamed tool-call fragments by `index`; parses arguments once at the end.
- Retries connect/5xx/429 errors with exponential backoff and **full jitter**, honouring
  `Retry-After` (capped at 10 s).
- **Never retries after the first token**; that would duplicate text.
- Classifies errors into `RateLimitError`, `ProviderUnavailable`, `ContextOverflow`, or a
  generic `ProviderError`, each carrying `retryable` and optional `retry_after_ms`.

`MockProvider` is deterministic: first pass requests `kb_search` with the user text, then
composes an answer from tool results or context. `mock_ttft_ms` and `mock_token_delay_ms`
control pacing for benchmarks.

`FallbackChain` walks models in order, but only before any token is emitted. Once output
exists, errors surface to the client as a terminal `error` frame.

## Observability

### Spans

One root span per connection (`ws.connection`) and one per turn (`turn`). Children:
`retrieval`, `retrieval.bm25`, `retrieval.dense`, `tool.call`, `audio.transcribe`,
`kb.ingest`, `kb.embed`, `kb.upsert`. Turn attributes include `ttft_ms`, `total_ms`,
`finish_reason`, `tool_iterations`, `kb_version`, `retrieval_cache_hit`, `degraded`, and
candidate counts.

`TraceRecorder` keeps a bounded LRU (200 traces) of the same trees and exposes them at
`GET /api/traces` and `GET /api/traces/{trace_id}`, so a demo does not need a collector.
When `otel_exporter=otlp` or `console`, the same spans also go to a real `TracerProvider`.
Default is `none`: spans are recorded in-process only.

### Metrics

`Metrics` is dependency-free: counters, gauges, and bounded histograms (`ttft_ms`,
`turn_total_ms`, `retrieval_ms`, `bm25_ms`, `dense_ms`, `fusion_ms`, `tool_call_ms`,
`ws_send_ms`). `GET /api/metrics` returns the merged fleet snapshot plus backend identities
and derived cache hit rate. Counters of note: `delta_frames_sent`, `delta_tokens_sent`,
`delta_frames_dropped`, `control_frames_waited`, `provider_retries`, `provider_fallbacks`,
`turns_aborted`, `ws_bad_frames`, `ws_stale_closed`, `ws_sessions_resumed`,
`ws_rejected_rate_limit`, `ws_rejected_connection_cap`, `rate_limit_cache_errors`,
`turn_checkpoints_dropped`, `session_journal_spooled`, `session_journal_replayed`,
`session_journal_cache_spooled`, `session_journal_cache_replayed`,
`session_journal_durability_degraded`, `turns_rejected_turn_budget`,
`turns_rejected_token_budget`, `turns_rejected_journal_unavailable`.

Each replica publishes a bucket snapshot to `cuepilot:metrics:{instance_id}` every
`metrics_publish_interval_s` (10 s) with a TTL of `metrics_publish_ttl_s` (30 s).
`/api/metrics` merges every live snapshot: counters and additive gauges are summed, and
histogram buckets are merged additively, so every observation is counted; percentiles are
interpolated within the fixed bucket layout. The `aggregation` block names the instances,
whether the response is merged, and the quantile method (`cumulative_buckets`) and bucket
bounds. `GET /metrics` renders the same merged snapshot in Prometheus text format. The
merge is pure and unit-tested (`telemetry/aggregate.py`); it is a stopgap for a real
metrics backend, not a replacement.

## Data model

```sql
CREATE TABLE kb_chunks (
  id         TEXT PRIMARY KEY,
  doc_id     TEXT NOT NULL,
  title      TEXT NOT NULL DEFAULT '',
  content    TEXT NOT NULL,
  embedding  VECTOR(1536) NOT NULL,
  meta       JSONB NOT NULL DEFAULT '{}',
  kb_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE TABLE kb_meta (id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1), version INT, updated_at timestamptz);
```

Chunk ids are `{doc_id}#{index}-{sha1(doc_id:index:piece)[:16]}`, so re-ingesting the same
document is idempotent. `kb_meta.version` is the cache generation. The embedding column is
fixed at 1536 dimensions; changing `CUEPILOT_EMBEDDING_DIM` requires altering the column and
re-ingesting (the memory store adapts automatically).

## Configuration

Every setting is `CUEPILOT_<UPPER_SNAKE>`; see [`.env.example`](../.env.example). Settings
load `.env` and `../.env`, so the server picks up the repo-root file when run from
`server/`. Tests construct `Settings` explicitly, so a developer `.env` cannot leak a real
key into CI.

Defaults chosen for the demo: hybrid retrieval, 8 hits from a 50-candidate pool, RRF k=60,
16 ms flush, 64-frame queue, 20 s heartbeat, 24-message history, 4 tool iterations,
OTel disabled-by-export.

## Tradeoffs and what changes at scale

- **In-process BM25.** Simple and fast for this corpus, but every replica builds its own
  index from the store at startup. At scale, move lexical search into Postgres
  (`tsvector`/`pg_trgm`) or OpenSearch and treat the store as the only index.
- **Exact search fallback.** `MemoryVectorStore` is O(n) per query. Fine for thousands of
  chunks, wrong for millions; it exists to keep the demo and tests hermetic.
- **Bucket-resolution cross-replica metrics.** Counters and gauges merge exactly; histogram
  percentiles are interpolated within fixed buckets, so they are not exact quantiles. At
  scale, push to Prometheus/OTLP and stop merging in-process.
- **Write-behind journal with a local spool and a cache fallback.** One worker drains the
  queue, so a slow Postgres costs checkpoints and, at worst, spills terminal writes to a
  capped local file and then to the shared cache. The cache fallback depends on Redis
  persistence. At scale the `turns` table needs partitioning by session and age, the spool
  should be a durable queue, and the journal should stream to an analytics sink rather than
  being read back one session at a time.
- **Bounded rate limiting.** Sliding-window counters plus per-replica socket cap, plus
  per-session turn and token budgets enforced post-hoc (a turn may overshoot by one
  `max_tokens`). Identity is socket-peer unless the operator names a trusted header or
  enables `X-Forwarded-For`. At scale, push to a real rate-limit backend.
- **No cross-encoder rerank.** RRF is cheap and robust; a reranker would improve precision
  on a larger corpus at the cost of a second model call.
- **Single active turn per session.** Simplifies cancellation and ordering. Concurrent
  turns would need per-turn outbound queues and a merge policy.

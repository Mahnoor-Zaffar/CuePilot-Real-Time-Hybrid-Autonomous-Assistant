"""Seed knowledge base for the demo.

A fictional-but-plausible internal KB for "CuePilot", the product this gateway
powers. Deliberately dense with identifiers, error codes and API field names so
the lexical layer has real work to do, plus prose passages that reward the dense
layer. `distractor_documents()` pads the corpus for honest retrieval benchmarks.
"""

from __future__ import annotations

from typing import Any

DOCUMENTS: list[dict[str, Any]] = [
    {
        "doc_id": "kb-streaming-protocol",
        "title": "Streaming protocol and frame types",
        "tags": ["streaming", "protocol"],
        "text": """
The gateway exposes a single full-duplex WebSocket at /ws/stream. Every frame is a
JSON object with a `type` discriminator and a protocol version negotiated in the
opening `hello` frame. Frames are intentionally small so that token deltas never
wait behind a large control payload.

Server-to-client frame types are `ready`, `ack`, `retrieval`, `tool_call_start`,
`tool_call_result`, `delta`, `done`, `aborted`, `error`, `pong` and `transcript`.
Client-to-server frame types are `hello`, `prompt`, `abort`, `audio`, `pong` and
`ping`. Unknown types are rejected with a non-retryable `bad_frame` error instead
of being silently ignored, which makes client/server version skew obvious.

`delta` frames carry an incrementing `seq` per turn. Clients must apply deltas in
sequence order and may discard duplicates after a reconnect. A `done` frame always
terminates a turn, even when the provider returned an error, so the client can
safely clear its in-flight skeleton state.
""",
    },
    {
        "doc_id": "kb-websocket-reconnect",
        "title": "WebSocket reconnect and backoff policy",
        "tags": ["resilience", "websocket"],
        "text": """
Clients reconnect with exponential backoff and full jitter: the delay for attempt
n is random(0, min(retry_backoff_cap_ms, base_ms * 2^n)). Defaults are base_ms=250
and retry_backoff_cap_ms=8000. Jitter is mandatory; without it every desktop client
in an office reconnects in lockstep and the gateway sees a thundering herd.

On reconnect the client sends `hello` with its previous `session_id`. If the
session is still inside the server-side TTL (default 3600 seconds) the gateway
restores conversation history and the current kb_version. If the session expired
the gateway returns a fresh session and the client re-renders from local state.

Half-open sockets are detected by application-level heartbeats every 20 seconds.
A connection that misses two consecutive heartbeat windows is closed server-side
so that stale connections do not hold Redis session leases.
""",
    },
    {
        "doc_id": "kb-error-codes",
        "title": "Gateway error code reference",
        "tags": ["errors", "reference"],
        "text": """
CUE-E1042 means the upstream model provider rejected the request because the
context window was exceeded. Fix: reduce retrieval_top_k, trim turn_history_limit,
or switch to a model with a larger context. The error is non-retryable without a
payload change.

CUE-E2041 is a provider rate limit (HTTP 429). The gateway retries with exponential
backoff and jitter, then walks the configured fallback model chain. If every model
is rate limited the turn fails with `retryable: true` and the client should wait at
least the value in the `retry_after_ms` field before resubmitting.

CUE-E3010 indicates the vector store returned no results while the corpus is
non-empty. It almost always means an embedding dimension mismatch between the
configured embedder and the pgvector column, for example a 1024-dimensional model
writing into a vector(1536) column.

CUE-E4021 is a client protocol violation: a frame arrived before `hello`, or the
`prompt` frame was missing its `id`. CUE-E5032 means the tool call exceeded
tool_timeout_s and was cancelled; the model is told the tool timed out and may
retry once with narrower arguments.
""",
    },
    {
        "doc_id": "kb-hybrid-retrieval",
        "title": "Hybrid retrieval: BM25, dense vectors and RRF",
        "tags": ["retrieval", "architecture"],
        "text": """
Retrieval runs two independent candidates generators in parallel. The lexical
layer is Okapi BM25 with k1=1.5 and b=0.75 over a code-aware tokenizer that splits
camelCase, snake_case and error codes. The semantic layer embeds the query and
searches pgvector using cosine distance with an HNSW index.

The two ranked lists are merged with Reciprocal Rank Fusion:
score(d) = sum over retrievers of 1 / (rrf_k + rank(d)), with rrf_k defaulting to
60. RRF is chosen over a cross-encoder because it adds no model latency at query
time and is insensitive to BM25 and cosine living on different score scales. The
fused result carries per-retriever ranks and contributions so the UI can explain
why a chunk surfaced.

Candidate pools default to retrieval_candidate_k=50 per retriever and are fused
down to retrieval_top_k=8. Retrieval results are cached in Redis for 120 seconds
under a key that includes kb_version, so an ingest invalidates results without any
explicit cache flush.
""",
    },
    {
        "doc_id": "kb-pgvector-operations",
        "title": "Operating the pgvector index",
        "tags": ["infra", "postgres", "pgvector"],
        "text": """
Chunks live in the kb_chunks table with a vector(1536) embedding column and an HNSW
index using vector_cosine_ops, m=16 and ef_construction=64. The default ef_search
of 40 is a good latency/recall trade-off; raise it per transaction with SET LOCAL
hnsw.ef_search = 100 when a query must maximise recall.

Ingest is idempotent: chunk ids are derived from doc_id, chunk index and a content
hash, and upserts use ON CONFLICT DO UPDATE. Every ingest increments kb_meta.version
in the same transaction, which is what makes retrieval cache keys self-invalidating.

Watch for table bloat after large re-ingests; HNSW indexes are not vacuumed
incrementally the way B-tree indexes are. Schedule VACUUM ANALYZE kb_chunks after
bulk loads, and prefer many small ingests over one huge replacement when possible.

If pgvector is unavailable the gateway falls back to an exact in-memory cosine
store when allow_degraded_infra is true. Recall is identical; the trade-off is
memory footprint and single-instance state.
""",
    },
    {
        "doc_id": "kb-session-state",
        "title": "Session and state management",
        "tags": ["sessions", "redis"],
        "text": """
Each WebSocket connection owns a Session object holding conversation history, the
scratchpad, the last kb_version seen and the active turn task. Sessions are hot in
process memory and mirrored to Redis so that a reconnect landing on another replica
can restore them.

History is capped at turn_history_limit=24 messages. Older turns are dropped from
the prompt but remain in the persisted session for the UI transcript. The cap is a
cost and latency control, not a storage limit.

Session writes are coalesced: the session is flushed to Redis on turn completion,
on abort, and on disconnect, rather than on every token. This keeps Redis writes at
roughly one per turn instead of thousands per turn. Session keys expire after
session_ttl_s so abandoned desktop clients cannot leak memory indefinitely.
""",
    },
    {
        "doc_id": "kb-cancellation",
        "title": "Cancelling in-flight generations",
        "tags": ["resilience", "streaming"],
        "text": """
An `abort` frame carries the turn id to cancel. The gateway resolves the active
turn task for that session and cancels it. Cancellation propagates through the
agent loop into the provider's async generator, which closes the upstream HTTP
connection, so a cancelled turn stops billing tokens immediately.

Submitting a new `prompt` while a turn is active implicitly aborts the previous
turn. The client receives an `aborted` frame for the old id before the `ack` for
the new one, which removes a whole class of UI races where two streams interleave.

Aborts are also published on the Redis event bus so that an abort routed to a
different replica still cancels the right task. Tool calls are shielded from
cancellation only long enough to release their resources; they never block the
abort for longer than 100 milliseconds.
""",
    },
    {
        "doc_id": "kb-rate-limits",
        "title": "Provider rate limits and fallback chain",
        "tags": ["resilience", "providers"],
        "text": """
The gateway treats 429 and 5xx responses as retryable before the first token is
emitted. Retries use exponential backoff with full jitter, honour the Retry-After
header when present, and stop after llm_max_retries attempts. Once the first token
has been emitted a retry would corrupt the stream, so mid-stream failures surface
as a `provider_stream_error` and the client decides whether to resubmit.

After retries are exhausted the gateway walks llm_fallback_models in order. A
fallback is only attempted if no tokens have been emitted for the current turn.
Every retry and fallback increments provider_retries and provider_fallbacks, which
are exported on /api/metrics.

If all models are rate limited the turn ends with CUE-E2041, `retryable: true` and
a retry_after_ms hint derived from the last Retry-After header or a 2-second
default.
""",
    },
    {
        "doc_id": "kb-tool-calling",
        "title": "Tool calling loop",
        "tags": ["llm", "tools"],
        "text": """
Tools are declared to the model with JSON Schema. The loop runs at most
max_tool_iterations=4 rounds; each round streams assistant text, collects tool
calls, dispatches them concurrently with asyncio.gather, appends the results as
tool messages, and continues. Tool dispatch is wrapped in tool_timeout_s=8 and a
failure is returned to the model as structured error text rather than raised.

The built-in tools are kb_search, which runs hybrid retrieval, and kb_open, which
returns the full text of a chunk id. Exposing kb_search as a tool lets the model
issue a second, narrower query when the first retrieval missed, which is the single
largest quality win in the whole pipeline.

Tool call frames are emitted to the client immediately: `tool_call_start` before
dispatch and `tool_call_result` after, so the UI can render an optimistic skeleton
and fill it in without waiting for the next text delta.
""",
    },
    {
        "doc_id": "kb-electron-hotkeys",
        "title": "Electron global hotkeys and contextBridge",
        "tags": ["electron", "desktop"],
        "text": """
The desktop client registers a global shortcut, CmdOrCtrl+Shift+Space, in the main
process using globalShortcut.register. The handler captures the focused window
title and the current clipboard selection, then sends them to the renderer over an
IPC channel named cue:context.

The renderer never gets Node.js access. contextIsolation is true, nodeIntegration is
false, and sandbox is true. A preload script exposes a narrow, typed API through
contextBridge: getContext(), onHotkey(), and captureAudio(). All IPC payloads are
validated in the main process before use.

If the global shortcut is already taken by another application, registration fails
silently on some platforms, so the main process checks the return value and surfaces
a non-blocking warning in the renderer.
""",
    },
    {
        "doc_id": "kb-audio-capture",
        "title": "Audio capture and PCM ingest",
        "tags": ["audio", "desktop"],
        "text": """
Audio is captured in the main process at 16 kHz mono, 16-bit little-endian PCM. The
renderer receives 40 millisecond frames, base64-encodes each frame, and sends it as
an `audio` frame with a monotonically increasing `seq`. Sending raw PCM rather than
encoded containers keeps the gateway free of codec dependencies.

The gateway buffers frames per session into a rolling window of audio_window_ms,
default 3000 milliseconds. When the window fills, it is submitted for transcription
and the buffer keeps only the trailing overlap so words are not cut at boundaries.
An `audio_ack` frame reports buffered_ms and rms so the client can render a level
meter without any additional round trip.

If transcription is disabled the gateway still accepts audio frames and returns
acks, which lets the client ship the capture path before the model endpoint is
provisioned. Audio frames are capped at ws_max_message_bytes; oversized frames
close the connection with CUE-E4021.
""",
    },
    {
        "doc_id": "kb-transcription",
        "title": "Whisper transcription pipeline",
        "tags": ["audio", "whisper"],
        "text": """
Transcription posts the rolling PCM window to an OpenAI-compatible
/audio/transcriptions endpoint using transcription_model, default whisper-1. The
response is streamed back to the client as `transcript` frames tagged partial or
final, so the UI can render an interim caption that firms up.

Partial transcripts are produced by re-transcribing the overlapping window and
diffing against the previous final transcript; the longest common prefix is kept
and the remainder is emitted as partial. This avoids a dedicated streaming ASR
dependency while still feeling live.

Latency budget: a 3 second window typically transcribes in 300 to 700 milliseconds,
so captions trail speech by roughly one window. If the provider is slow the gateway
drops intermediate windows rather than queueing them, because stale captions are
worse than missing ones.
""",
    },
    {
        "doc_id": "kb-telemetry",
        "title": "OpenTelemetry spans and TTFT measurement",
        "tags": ["observability", "otel"],
        "text": """
Every turn opens a root span named turn with attributes session_id, turn_id,
kb_version and model. Child spans cover retrieval.bm25, retrieval.dense,
retrieval.fusion, llm.first_token, llm.stream, tool.call and ws.send. Spans are
exported through an OTLP/HTTP exporter when otel_exporter=otlp, or printed to the
console in development.

Time to First Token is measured from the moment the prompt frame is decoded to the
moment the first delta is handed to the socket writer, not to when the provider
returns its first byte. That definition includes queueing and retrieval, which is
what the user actually experiences.

The in-memory trace buffer keeps the last 200 trace trees and serves them at
/api/traces/{trace_id}. This makes latency regressions reproducible in a demo
without running a collector. TTFT, retrieval and tool-call histograms are exposed
as p50/p95/p99 at /api/metrics.
""",
    },
    {
        "doc_id": "kb-deployment",
        "title": "Deployment topology and scaling",
        "tags": ["infra", "deployment"],
        "text": """
The gateway is stateless apart from hot session caches, so it scales horizontally
behind a load balancer that supports WebSocket upgrade and long idle timeouts.
Sticky sessions are recommended but not required because session state is mirrored
to Redis and abort events fan out over pub/sub.

Each replica keeps one asyncpg pool sized max_size=10 and one Redis connection pool.
With 500 concurrent streams per replica, memory is dominated by the per-connection
send queue (ws_send_queue_size frames) and by in-flight prompt contexts, not by
Python object overhead.

Health checks hit /api/health, which reports database and Redis reachability and the
active provider mode. A replica that loses Redis but keeps Postgres continues serving
with degraded session persistence; it reports degraded: true rather than failing the
health check, so a Redis blip does not trigger a full rollout restart.
""",
    },
    {
        "doc_id": "kb-security",
        "title": "Authentication and prompt-injection controls",
        "tags": ["security"],
        "text": """
REST endpoints accept an optional Bearer token configured via api_token; WebSocket
clients pass the same token as the `token` query parameter. When api_token is unset
the gateway is open, which is only acceptable on localhost development.

Retrieved chunks are treated as untrusted content. They are inserted into the prompt
inside an explicit context block that instructs the model to treat them as data, and
tool calls requested by retrieved text are not auto-approved. Only the tool registry
can dispatch, and each tool validates its arguments against a schema.

The Electron renderer runs with contextIsolation and sandbox enabled and a strict
Content-Security-Policy that forbids remote script loading. Audio and clipboard data
never leave the machine unless the user explicitly submits a turn.
""",
    },
    {
        "doc_id": "kb-billing",
        "title": "Usage metering and cost controls",
        "tags": ["billing"],
        "text": """
Every turn records prompt_tokens, completion_tokens and cached_tokens from the
provider's usage block. Cost is computed with a per-model price table and written to
the usage ledger keyed by workspace_id and day. Streams cancelled mid-flight are
billed for the tokens actually generated, which is why cancellation closes the
upstream connection promptly.

Hard caps are enforced before dispatch: if a workspace has consumed its monthly
budget the turn fails fast with CUE-E6020 instead of spending more. Soft caps emit a
warning frame but allow the turn to proceed.

Retrieval is free at the margin, so the cost lever that matters is context size:
retrieval_top_k and turn_history_limit together dominate prompt tokens. Caching
retrieval results also cuts embedding spend on repeated questions.
""",
    },
    {
        "doc_id": "kb-sdk-python",
        "title": "Python SDK quickstart",
        "tags": ["sdk", "python"],
        "text": """
The Python SDK wraps the WebSocket protocol in an async context manager. Install
with pip install cuepilot, then connect with `async with CuePilotClient(url, token)`
and iterate over events from `client.ask("...")`. Each event has a `kind` attribute
matching the wire frame type.

Aborts are handled by breaking out of the async for loop; the SDK sends the abort
frame and waits for the terminal frame before returning. Reconnects are automatic
with the same jittered backoff as the desktop client, and the SDK replays the last
acknowledged seq so no tokens are duplicated.

The SDK requires Python 3.11 or newer because it uses asyncio.TaskGroup internally
for concurrent tool result collection.
""",
    },
    {
        "doc_id": "kb-sdk-typescript",
        "title": "TypeScript SDK and React hook",
        "tags": ["sdk", "typescript", "react"],
        "text": """
The TypeScript SDK exposes `createCueStream()` returning an object with `send`,
`abort` and an async event iterator. It is transport-agnostic: pass a WebSocket
factory so the same client works in Electron, the browser and Node tests.

The bundled React hook `useCueStream` batches token deltas with
requestAnimationFrame. Deltas arriving between frames are concatenated into one
string and applied once per frame, which caps React renders at roughly 60 per second
instead of one per token. Under a 90 token/second stream this cuts render work by
about 33 percent and eliminates main-thread jank on long answers.

The hook also exposes `pendingToolCalls` so the UI can render skeletons optimistically
before the server acknowledges the call.
""",
    },
    {
        "doc_id": "kb-frontend-performance",
        "title": "Renderer performance and frame budgeting",
        "tags": ["frontend", "performance"],
        "text": """
The renderer targets a 16 millisecond frame budget. Token deltas are appended to a
buffer and flushed inside requestAnimationFrame; if a frame is already pending the
new text is merged into it. This is the client half of backpressure: the server
coalesces deltas into ~16 millisecond batches, and the client never renders faster
than the display refresh.

Markdown rendering is the expensive part of a long answer. The renderer parses only
the trailing unclosed block during streaming and re-parses the full document once on
`done`. Syntax highlighting is deferred until the turn completes.

Idle memory target is under 50 MB for the renderer process. Avoid retaining full
transcripts in component state; keep the last N turns and page older ones from disk.
""",
    },
    {
        "doc_id": "kb-backpressure",
        "title": "Backpressure and frame coalescing",
        "tags": ["streaming", "performance"],
        "text": """
Each connection owns a bounded send queue of ws_send_queue_size frames. The agent
pushes deltas into a coalescer, not directly into the socket. The coalescer merges
all deltas that arrive inside a delta_flush_ms window, default 16 milliseconds, into
a single delta frame with a concatenated text field.

Control frames such as tool_call_start and done bypass the coalescer and take
priority, so a burst of tokens can never delay a terminal frame. If the queue is
full, delta frames are dropped and counted in delta_frames_dropped, while control
frames block the producer until there is room.

This keeps outbound frame rate near 60 per second regardless of token rate. A
90 token/second stream produces roughly 60 frames per second instead of 90, and a
600 token/second burst produces the same 60 rather than melting the client.
""",
    },
]


def distractor_documents(count: int = 36) -> list[dict[str, Any]]:
    """Unrelated-but-lexically-similar docs to make retrieval non-trivial."""
    topics = [
        ("billing invoices", "invoice", "payment method, proration, tax id, dunning"),
        ("onboarding checklist", "onboarding", "invite teammates, sso setup, workspace roles"),
        ("marketing attribution", "attribution", "utm parameters, campaign ids, conversion windows"),
        ("mobile offline mode", "mobile", "local cache, conflict resolution, sync queue"),
        ("data export", "export", "csv export, ndjson, signed download url, retention"),
        ("webhook delivery", "webhook", "signature header, retries, dead letter queue"),
        ("search analytics", "analytics", "query volume, zero result rate, click through"),
        ("accessibility", "a11y", "focus order, aria labels, reduced motion"),
        ("localization", "i18n", "locale files, plural rules, right to left"),
        ("email deliverability", "email", "spf, dkim, dmarc, bounce handling"),
        ("feature flags", "flags", "percentage rollout, targeting rules, kill switch"),
        ("audit logging", "audit", "actor, action, resource, immutable retention"),
        ("backup restore", "backup", "snapshot cadence, point in time recovery, rpo"),
        ("incident response", "incident", "severity levels, on call rotation, postmortem"),
        ("capacity planning", "capacity", "headroom, p99 load, autoscaling policy"),
        ("api versioning", "versioning", "deprecation window, sunset header, changelog"),
        ("key rotation", "keys", "rotation cadence, dual key overlap, revocation"),
        ("search ranking", "ranking", "relevance signals, freshness boost, personalization"),
        ("file storage", "storage", "object storage, presigned upload, virus scan"),
        ("notification preferences", "notifications", "digest cadence, quiet hours, channels"),
        ("tenant isolation", "tenancy", "row level security, schema per tenant, noisy neighbor"),
        ("pricing tiers", "pricing", "seat based, usage based, overage, annual commitment"),
        ("support escalation", "support", "severity matrix, response sla, escalation path"),
        ("release process", "release", "canary, blue green, rollback criteria"),
        ("data residency", "residency", "region pinning, replication, cross border transfer"),
        ("model evaluation", "evals", "golden set, judge model, regression threshold"),
        ("prompt caching", "caching", "prefix reuse, cache hit rate, ttl"),
        ("vector index tuning", "tuning", "ef search, recall, quantization trade-offs"),
        ("queue workers", "workers", "concurrency, visibility timeout, poison messages"),
        ("schema migrations", "migrations", "expand contract, backfill, online ddl"),
        ("cost allocation", "cost", "tagging, showback, anomaly detection"),
        ("customer health score", "health", "usage trend, support tickets, renewal risk"),
        ("document ingestion", "ingestion", "parsing, ocr, deduplication, chunking"),
        ("semantic caching", "semantic", "near duplicate queries, similarity threshold"),
        ("traffic shaping", "shaping", "rate limit buckets, burst allowance, fairness"),
        ("disaster recovery", "dr", "failover region, rto, game day drills"),
    ]
    docs: list[dict[str, Any]] = []
    for index in range(count):
        title, slug, keywords = topics[index % len(topics)]
        docs.append(
            {
                "doc_id": f"kb-distractor-{slug}-{index:02d}",
                "title": f"{title.title()} runbook ({index:02d})",
                "tags": ["distractor"],
                "text": (
                    f"This runbook covers {title} for the platform team. It documents "
                    f"{keywords}. The primary owner rotates quarterly. Escalate to the "
                    f"platform on-call if a {slug} change is expected to affect more than "
                    f"five percent of traffic.\n\n"
                    f"Operational notes: review the {slug} dashboard weekly, keep the "
                    f"documented thresholds current, and record every exception with a "
                    f"link to the change ticket. Dashboards are grouped by workspace and "
                    f"region so that a single noisy tenant is visible immediately.\n\n"
                    f"Change procedure for {slug}: open a change ticket, request review "
                    f"from the owning team, roll out to the staging workspace first, and "
                    f"watch error budget burn for one hour before promoting to production. "
                    f"Rollback is the previous known-good configuration, which is retained "
                    f"for thirty days."
                ),
            }
        )
    return docs


def all_documents(include_distractors: bool = True) -> list[dict[str, Any]]:
    return DOCUMENTS + (distractor_documents() if include_distractors else [])

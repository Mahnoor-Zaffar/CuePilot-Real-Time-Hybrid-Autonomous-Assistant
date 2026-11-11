# Failure modes

Real-time AI fails in ordinary ways: sockets drop, providers rate limit, queues fill, caches
go stale, infrastructure disappears. This document records each failure this repo handles,
how it is detected, what the user sees, and where the code lives.

The design rule throughout: **degrade the feature, not the turn** — and report the
degradation instead of hiding it.

## Summary

| Failure | Detection | Behavior | Recovery |
|---|---|---|---|
| Gateway unreachable | socket close/error | jittered reconnect banner | automatic, ≤8 attempts |
| Server restart mid-turn | close during stream | reconnect + resume + replay; partial turn marked `interrupted` | automatic |
| Connection flood | sliding-window counter + per-replica gauge | `client_rate_limited` / `connection_cap` + close `4429` | after `retry_after_ms` |
| Provider 429 | HTTP 429 / SSE error | retry with backoff → model fallback → `rate_limited` | user retries |
| Provider 5xx / transport | status/exception | same as 429, then `provider_unavailable` | user retries |
| Provider dies mid-stream | exception after first token | **no retry**, terminal `error` | user retries |
| Context overflow | provider error text | `context_overflow` (not retryable) | shorten prompt |
| Tool timeout | `asyncio.wait_for` | tool result `ok:false`, loop continues | model adapts |
| Unknown/invalid tool | registry lookup / validation | error result to the model | model adapts |
| One retriever fails | gather exception | fuse the other; `degraded` list | next query |
| Both retrievers fail | empty ranked set | answer without context, `degraded` | next query |
| Postgres down at boot | probe timeout | in-memory vector store; sessions mirror-only (`journal.backend = "none"`) | automatic |
| Schema drift / bad DDL | checksum + DDL error | **startup fails**, no degraded fallback | fix the migration |
| Embedding width changed | recorded render params | rebuild owning migration via `.down.sql`, bump KB version; journal untouched | automatic |
| Postgres lost mid-run | query exception | dense degrades to BM25; journal writes retry, then count as failed | reconnect |
| Redis down | probe timeout | in-process cache/bus; resume reads the journal | automatic |
| Journal queue full | `QueueFull` on enqueue | checkpoints drop (`journal.dropped`); terminal turns spool to local disk, then to the shared cache, then reject new prompts (`journal_unavailable`) | replay loop |
| Turn journal write fails | exception from the drain worker | retry once; terminal turns spool, then replay | replay loop |
| Hard crash mid-turn | stale `streaming` row / inflight key on read | partial answer restored, marked `interrupted`, with `checkpointed_at` timestamp | automatic |
| Spool full / unwritable | `session_journal_spool_full` / `_errors` | terminal write falls back to the shared cache; if that also fails, `durability_degraded` is set and new prompts are refused with retryable `journal_unavailable` | operator frees disk / Redis |
| Metrics across replicas | per-instance snapshots in cache | merged counters/gauges; cumulative bucket histograms merged additively; Prometheus `/metrics` | next publish |
| KB ingest during session | version bump | new cache key + bus invalidation | next turn |
| Slow client / full queue | `QueueFull` | drop **frames**, keep text | automatic |
| Wedged socket | 5 s send timeout | close connection | client reconnects |
| Silent client | no frame for 2.5× heartbeat | close 4408 | client reconnects |
| Malformed frame | Pydantic validation | `bad_frame`, socket stays open | client fixes |
| Bad handshake | timeout / wrong frame | `error` + close 4400 | client fixes |
| Bad token | query param check | `error` + close 4401 | client fixes |
| Unknown session id | cache miss, then journal miss | new session (no replay) | client replaces id |
| Abort mid-stream | client `abort` frame | task cancel, flush text, `aborted` | user continues |
| Mic denied / no Worklet | browser exception | inline audio error, text still works | grant permission |
| Screen Recording denied | `getMediaAccessStatus("screen")` | capability IPC returns a reason; no capture starts | grant permission + restart |
| Loopback unsupported (Linux) | platform check | `systemAudio: false` with a reason | use mic or another OS |
| Loopback returns no audio track | empty `getAudioTracks()` | stop the stream, show a permission hint | retry after granting |
| Transcription fails | HTTP exception | log + metric, no transcript | best effort |

## Reconnect backoff

Implemented in `desktop/src/useCueStream.ts`.

```
ceiling = min(8000, 250 * 2^attempt)
delay   = max(250, random() * ceiling)     // full jitter
```

- `attempt` increments on every failed connection, capped at 8.
- Full jitter (random over the whole ceiling, not ceiling/2) avoids synchronized retry
  storms when a gateway restarts and many clients reconnect at once.
- A successful `ready` resets `attempt` to 0.
- The UI shows `Reconnecting in Ns · last close <code>` so the wait is visible, not a
  spinner.
- The session id in `localStorage` (`cuepilot.session_id`) is sent in `hello`; the server
  replays the last 20 messages if it still knows the session.
- A deliberate "new session" close reconnects immediately instead of backing off.

**What is not handled:** an offline queue of unsent prompts. If the socket is closed,
`sendPrompt` returns false and the composer keeps the draft. Persisting a client-side
outbox would be the next step.

## Cancellation

The client sends `{type:"abort", id}` (or omits `id` for the active turn). The connection
cancels the turn task and waits up to 2 s for it to unwind. Inside `_run_turn`,
`CancelledError` triggers `task.uncancel()` before the post-cancel `await`s; without that,
those awaits would immediately re-raise and the `aborted` frame would never be sent. Buffered
text is flushed first, then `aborted` with reason `cancelled` (or `superseded` when a new
prompt replaced it).

The provider stream lives inside `async with client.stream(...)`; cancellation closes the
upstream response, so the provider stops generating and billing. A turn already past its
`done` frame has no active task, and abort replies `no_active_turn` for a named id.

Cross-replica abort: if the `id` does not match the local active turn, the connection
publishes `turn.abort` on the bus and answers `aborted` with reason `not_local`; the replica
owning the socket cancels the matching task. Without Redis, this fan-out is unavailable and
the abort only affects the local process.

**Not handled:** an abort racing the final token. The turn may finish with `done` just
before the cancel lands; the client treats the first terminal frame as authoritative and
ignores the later one.

## Server restart mid-turn

Symptom: the stream stops, the socket closes with `1006`, and the turn never receives
`done`.

Behavior: the client marks the turn not-streaming only when a terminal frame arrives; on
close it schedules a reconnect. After `ready`, if the server still has the session (Redis
mirror) it sends `history`, which **replaces** the local transcript. With Postgres available,
the partial answer is there: a streaming turn is checkpointed every
`turn_checkpoint_interval_ms` (250 ms), so a hard crash loses at most one interval of text.
History replay flags that turn with `interrupted: true`, and the client can render it as a
partial answer instead of pretending it finished.

Recovery of the row is lazy, on the read path: `load_session` marks a `streaming` row older
than `turn_checkpoint_stale_s` (60 s) as `aborted` / `interrupted` before replaying it. A
fresh `streaming` row is left alone in case another replica is still producing it, so no
separate janitor is required.

Tradeoff: checkpoints are droppable and terminal writes are not. A checkpoint is
reconstructible from the next one, so under queue pressure it is shed; a terminal turn is the
transcript, so it spools to local disk rather than disappearing. What resume still does not
replay is the in-flight token buffer: the client gets the last checkpointed prefix, not the
exact frame boundary.

## Provider failures

### Rate limit (429)

`OpenRouterProvider._classify` returns `RateLimitError` with `retry_after_ms` parsed from
the `Retry-After` header. The provider retries up to `llm_max_retries` (3) using
`min(retry_after, 10s)` when supplied, otherwise exponential backoff with full jitter.

If retries are exhausted before any token, `FallbackChain` moves to the next model in
`llm_fallback_models` and increments `provider_fallbacks`. If every model fails, the turn
ends with:

```json
{ "type": "error", "code": "rate_limited", "retryable": true, "retry_after_ms": 4000 }
```

The client shows the error on the assistant message; `retryable` is available for a future
"Retry" affordance. The socket stays open.

### 5xx / transport failure

Classified as `ProviderUnavailable` (`retryable: true`), handled by the same retry and
fallback path. A connect or read timeout before the first token is retried; after the first
token it is not.

### Failure after the first token

**Never retried.** Retrying would duplicate or reorder text. The agent closes the upstream
stream, emits a terminal `error` with the provider's code, and emits `done` with
`finish_reason: "error"`. Metrics: `provider_errors_<code>`, `turns_failed`.

The user keeps the partial answer; it is visually marked with the error. This is deliberate:
silently discarding half an answer is worse than showing it with a warning.

### Context overflow

Detected by matching `context` + (`length`|`window`) in the provider body. Emitted as
`context_overflow`, `retryable: false`. The gateway does not silently truncate history
because that changes semantics; the caller decides. `turn_history_limit` (24) and
`retrieval.to_context(max_chars=6000)` bound the common case.

## Tool failures

`ToolRegistry.dispatch` runs every call under `tool_timeout_s` (8 s) and converts any
exception into a `ToolResult(ok=False)` that is fed back to the model. The model can retry
with different arguments. A tool failure never fails the turn.

- Unknown tool → `unknown tool \`name\``.
- Invalid arguments → validation message (`query must be a non-empty string`, etc.).
- `kb_open` with an unknown chunk → validation error.
- Timeout → `tool timed out after 8.0s`.
- Multiple calls run concurrently with `asyncio.gather(return_exceptions=True)`; one failure
  does not cancel the others.

Loop bound: `max_tool_iterations` (4). Exceeding it sets
`finish_reason: "tool_iterations_exhausted"` and returns the last text, so a model that
keeps calling tools cannot hang the turn.

## Retrieval failures

The orchestrator launches BM25 and dense concurrently and uses
`asyncio.gather(..., return_exceptions=True)`:

- **One fails** → the failure is recorded in `degraded`, `retrieval_<name>_failures`
  increments, and the surviving list is fused. The UI shows a "degraded" badge on the
  source panel.
- **Both fail** → empty hits, `degraded: ["bm25","dense"]`, and the agent answers from
  conversation history only. It does not fabricate that context exists; the system prompt
  says the context is insufficient.
- **Retrieval raises outside the gather** (unexpected) → the agent logs it and proceeds with
  `degraded: ["retrieval"]`.

Per-retriever timeouts are `retrieval_timeout_s` (6 s), applied to the embedding call and
the vector query. BM25 runs in a worker thread so it cannot block the event loop.

## Stale context invalidation

This is the subtle one: retrieval caches and BM25 indexes must not outlive the data.

1. Ingest writes chunks, then `store.bump_version()` increments `kb_meta.version`.
2. `HybridRetriever.refresh()` rebuilds BM25 from the store and adopts the new version.
3. Every cache key embeds `kb_version`, so all prior results become unreachable without an
   explicit flush (`cuepilot:ret:{version}:{digest}`).
4. The ingest endpoint publishes `kb.invalidate`; other replicas call `refresh()` and skip
   their own origin event.
5. The `ready` frame carries `kb_version`, so the client can display which generation it is
   talking to.

In-flight turns keep the context they already retrieved; the next turn sees the new
version. That is the correct boundary: mutating context mid-generation would make citations
inconsistent with the answer.

If Redis is unavailable, the bus falls back to in-process; invalidation then only reaches
the current process. This is reported in `/api/health` as a degraded bus.

## Infrastructure loss

### Postgres unavailable at startup

`build_store` probes with a short timeout. On failure (and
`allow_degraded_infra=true`) it returns `MemoryVectorStore` and a reason string. Startup
continues, `seed_on_start` ingests the corpus into memory, and health reports
`"status": "degraded"`. Retrieval quality is unchanged for this corpus. Durability is lost:
there is no journal, so `journal.backend` is `"none"` and sessions live only in Redis.

### Schema migration failure

`build_store` applies pending migrations before the store is usable. The runner holds
`pg_advisory_lock(hashtext('cuepilot:migrations'))` for the whole run, so replicas that boot
together serialise instead of racing; the losers find the work already recorded.

Startup fails — it does **not** degrade to the memory store — when:

- an already-applied file's checksum changed (someone edited history): add a new migration;
- a parameter changed and the migration has no `.down.sql` (pgvector cannot cast between
  vector widths, so a width change must be a rebuild);
- the DDL itself fails, for example `CREATE EXTENSION vector` without privileges.

A parameter change is expected, not an error. The runner applies the down file (drops the
derived `kb_chunks`, bumps `kb_meta.version` so retrieval cache entries keyed on the old
vectors miss), re-runs the file at the new width, and boot re-ingests the corpus. The first
boot after a switch costs one ingest; nothing durable is lost because `kb_chunks` is derived.

Only the migration that **declares** a parameter rebuilds, and a rebuild cascades only
through explicit `-- requires:` headers. The journal (`002`) declares neither, so changing
the embedding width rebuilds `001` and leaves `sessions`/`turns` intact — this is covered by
a Postgres test. The one hazard the runner cannot see is an undeclared dependency: a file
that alters an earlier migration's objects without the header will not be re-applied, which
is why the header is part of review.

### Postgres lost after startup

`PgVectorStore` queries raise. Dense retrieval is marked degraded and BM25 keeps serving
(the index is in memory). BM25 refreshes will fail until Postgres returns; the existing
index remains usable. Health only reflects the boot-time probe, so a runtime loss is visible
through `retrieval_dense_failures` and the `degraded` field on `retrieval` frames.

### Session journal

`SessionManager` writes Redis synchronously and enqueues the Postgres write on a bounded
queue drained by one worker. That split is the whole design: Redis is the fast path, Postgres
is the record, and the token path never awaits durability.

- **Checkpoints.** A streaming turn is enqueued every `turn_checkpoint_interval_ms` (250 ms)
  with `status = "streaming"`. Checkpoints are best-effort: when the queue is full they are
  dropped and counted (`turn_checkpoints_dropped`), because the next checkpoint carries the
  whole answer so far. A hard crash loses at most one interval.
- **Terminal turns.** `done`, `aborted` and `error` writes are the transcript. When the queue
  is full, or the worker's write fails after its retry, a terminal record is appended to a
  local NDJSON spool instead of being dropped. The spool is replayed by a background task
  every `session_journal_spool_retry_s` (30 s) and once more during graceful shutdown.
- **The spool is bounded.** `session_journal_spool_max_bytes` (8 MB) caps it; when full,
  `session_journal_spool_full` increments and the terminal write is lost. That is the honest
  limit: local disk is a buffer, not a second database, and it does not survive losing the
  container filesystem.
- **A turn lands before its session row.** Under backpressure the session upsert can be
  dropped while the turn write survives. The turn write first does
  `INSERT INTO sessions (id) VALUES ($1) ON CONFLICT DO NOTHING` so the foreign key never
  rejects a transcript line.
- **Graceful shutdown** drains for `session_journal_drain_s` (3 s), then gives the spool one
  bounded replay attempt before cancelling both tasks. A hard kill loses at most the queue
  and whatever the spool had not yet flushed.

Health exposes
`journal: {backend, queued, written, dropped, failed, spool: {enabled, path, spooled, replayed, errors}}`.
With no Postgres the journal is absent (`backend = "none"`) and sessions are mirror-only;
resume then depends on Redis still holding the key.

### Redis unavailable

Cache and bus both fall back to in-process implementations. Consequences:

- Retrieval caching still works, per process.
- Session resume still works: hot dict first, then the Postgres journal. Only with no
  Postgres either does an evicted session become a new session.
- Cross-replica `kb.invalidate` and `turn.abort` no longer reach other replicas.

Health and `/api/metrics` report `backends.cache = "memory"` and `backends.bus = "memory"`.

## Connection limits

Two limits run before a socket is admitted (`app/ws/limits.py`):

1. **Per-client attempt rate.** A sliding-window counter keyed by
   `cuepilot:ratelimit:connect:{identity}` caps opens at
   `rate_limit_connect_attempts` (30) per `rate_limit_window_s` (60 s). The window is a
   sorted set in Redis (server `TIME`) and a monotonic deque in memory, so there is no
   fixed boundary to burst across. With Redis the counter is cross-replica; with the
   in-process cache it is per-replica, and `limits.backend` in health says which.
2. **Per-replica connection cap.** `rate_limit_max_connections` (200) bounds this process's
   sockets and turn tasks. It is local on purpose: `ws_active` is a process gauge.

The cap check and the gauge increment are separated by an `await`, so the limiter takes a
**synchronous reservation** before its first await and releases it once the socket is
accepted. Without that, two simultaneous opens could both pass a check against the same
count.

A rejected client receives an `error` frame (`client_rate_limited` or `connection_cap`,
`retryable: true`, `retry_after_ms`) and the socket closes with `4429`, so it can back off
instead of reconnect-looping.

The limiter **fails open** if the cache raises: it logs, increments
`rate_limit_cache_errors`, and admits the connection. A broken limiter must not take the
gateway down. The failure is visible in metrics and in the health `limits` block.

Residual caveats: identity is the socket peer unless the operator names a trusted header
(`rate_limit_identity_header`) or enables `rate_limit_trust_forwarded_for` behind a trusted
proxy — clients behind one NAT still share a bucket by default. The sliding window removes
the boundary burst but remains a counter, not a leaky bucket. Turn and token budgets are
enforced per session over the same window; the token budget is post-hoc and may overshoot
by at most one `llm_max_tokens`.

## Cross-replica metrics

`/api/metrics` used to report whichever worker answered the request. Now each replica
publishes a snapshot to `cuepilot:metrics:{instance_id}` every
`metrics_publish_interval_s` (10 s) with a `metrics_publish_ttl_s` (30 s) TTL, and the
endpoint merges every live snapshot (`app/telemetry/aggregate.py`):

- **Counters are summed.** A replica restart resets its own counters, so a merged total can
  move backwards. `aggregation.instances` and `aggregation.instance_ids` say what was
  visible.
- **Gauges are summed.** Every gauge published today is additive (`ws_active`); a
  non-additive gauge would need an explicit rule.
- **Histogram buckets are merged additively.** Each worker publishes the cumulative bucket
  counts for every histogram, so every observation is counted even when a worker observed
  far more than a sample cap would have allowed. Percentiles are interpolated within the
  fixed bucket layout and are therefore bucket-resolution, not exact. The bucket bounds and
  quantile method (`cumulative_buckets`) are reported in `aggregation`.
- If the cache cannot be read, the endpoint still returns local metrics and adds
  `aggregation.error`; it does not fail the request.
- A graceful shutdown deletes its own snapshot so a restart does not linger as a phantom
  second instance for the rest of the TTL.

The response always includes `aggregation: {backend, instances, instance_ids, merged,
quantile_method, bucket_bounds_ms, note}` so a caller can tell a single-process answer
from a fleet answer. `GET /metrics` renders the same merged snapshot in Prometheus text
format.


## Backpressure and slow clients

Three independent limits prevent one slow socket from exhausting the server:

1. **Coalescing.** Deltas are merged every 16 ms. At 90 tok/s that is ~2 tokens per frame;
   the outbound frame rate is bounded near 60/s no matter how fast the model talks.
2. **Bounded queue.** `ws_send_queue_size` (64). When full, a delta frame is discarded and
   its text is re-inserted at the head of the turn buffer; the next flush resends it.
   Metrics: `delta_frames_dropped`. **Text is never lost by this path.**
3. **Buffer cap.** `MAX_BUFFER_CHARS` (64k) per turn. Only if the client is truly gone does
   the oldest text get dropped; `delta_chars_dropped` records the character count.

Control frames (`done`, `error`, `aborted`, tool events) are never dropped: `send()` waits
up to 3 s for space (`control_frames_waited`) and closes the connection if the queue is
wedged. Before a terminal frame the turn buffer is flushed, preserving order.

Client side, `requestAnimationFrame` batching bounds React updates to one per frame. If the
window is backgrounded and rAF is throttled, text accumulates in a pending map and is
applied when the tab is visible again; it is not dropped.

## Socket and protocol failures

- **Silent client.** The receive loop waits `2.5 × ws_heartbeat_interval_s` (50 s). On
  timeout it closes `4408` and increments `ws_stale_closed`. The server also pings every
  20 s so a dead peer is detected even if it sends nothing.
- **Wedged writer.** Each `send_json` has a 5 s timeout; failure sets the stream closed and
  the connection tears down (`ws_send_failures`).
- **Malformed frame.** Schema errors produce `bad_frame` and the socket stays open
  (`ws_bad_frames`), so a single bad message does not kill a session.
- **Bad handshake.** Missing `hello`, non-JSON, or a version mismatch produce an `error`
  frame and close `4400`. Auth failure closes `4401`.
- **Unknown session id.** `resume_or_create` creates a new session and returns
  `resumed: false`; the client replaces its stored id. No error is raised, because a stale
  id after a Redis flush is normal.

## Audio failures

- `audio_enabled=false` → every audio frame gets `audio_disabled`.
- Invalid base64 or PCM → `bad_audio_frame`; the socket stays open.
- Mic permission denied or `getUserMedia` unavailable → the hook sets `audio.error` and the
  UI shows it inline; text chat is unaffected.
- `AudioWorklet` unsupported → `addModule` throws, the same inline error path runs.
- Transcription is best effort: HTTP failures increment `transcription_failures` and are
  logged; no `transcript` frame is sent. The audio buffer keeps rolling.
- Buffer is bounded to 3 s with a trailing one-third overlap when consumed, so a slow
  transcription does not grow memory and words are not cut at the window edge.

## Measurement caveats

- **Clock skew.** Client TTFT is `Date.now()` at send → first painted delta, on one machine,
  so it is skew-free. RTT uses the echoed `ts` and clamps negatives to 0; it is still
  affected by wall-clock adjustments. Server TTFT uses `perf_counter`, which is monotonic.
- **Mock pacing.** Benchmarks set `mock_ttft_ms=120` and `mock_token_delay_ms=4`. Those
  numbers are a test harness, not a provider claim.
- **Degraded CI.** Tests point Postgres and Redis at `127.0.0.1:1`, so health is
  `"degraded"` and every fallback path is exercised on every run.

## Known gaps

- **Rate limiting is bounded, not exact.** The connection counter is a sliding window over
  the configured window, so there is no boundary burst, but it is still a counter, not a
  leaky bucket. Per-session turn and token budgets are enforced, but the token budget is
  post-hoc: a turn may overshoot by at most one `llm_max_tokens` because the cost is only
  known after the provider reports usage. The connection cap stays per-replica because
  `ws_active` is a process gauge. Identity is socket-peer unless the operator names a
  trusted header or enables `X-Forwarded-For` — an untrusted header is a spoofable key.
- **Durability is bounded, not guaranteed.** Checkpoints lose at most one interval on a
  hard crash, and terminal writes try local disk first, then the shared cache. The local
  spool is capped at 8 MB; the cache fallback depends on Redis persistence, which is the
  operator's responsibility. When both are unavailable, `durability_degraded` is set and
  new prompts are refused with retryable `journal_unavailable` until a write or replay
  succeeds. Losing the container filesystem no longer loses the spool, but it does lose the
  local copy of anything not yet replayed.
- **Merged percentiles are bucket-resolution.** Histograms merge cumulative buckets
  additively, so every observation is counted, but quantiles are interpolated within the
  fixed bucket layout (`docs/ARCHITECTURE.md`), not exact. Counters still reset when a
  replica restarts. Exact fleet-wide quantiles need a real metrics backend.
- **A dropped checkpoint is now visible.** The client renders a restored partial with its
  `checkpointed_at` timestamp, so the age of the partial is explicit. It is still not a
  frame-exact replay: a hard crash loses at most one checkpoint interval of text.
- System audio is macOS and Windows only, and on macOS it needs Screen Recording
  permission; a denied permission surfaces as a reason in the UI, not as silence. Linux
  depends on the sound server and is not claimed.
- The retrieval corpus is synthetic. Intervals quantify sampling noise on this label set,
  not fit to any real domain. An opt-in BEIR subset (`make beir`) provides an external
  reference point; its numbers are not the headline.

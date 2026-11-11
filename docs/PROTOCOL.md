# Wire protocol (v1)

One WebSocket at `/ws/stream`. Every frame is a UTF-8 JSON object with a `type`
discriminator. The server validates client frames with Pydantic and answers malformed input
with an `error` frame rather than closing silently.

- **Version:** `1`. `hello.protocol_version` must match; a mismatch closes with `4400`.
- **Auth (optional):** if `CUEPILOT_API_TOKEN` is set, connect with `?token=<token>`.
  A missing/wrong token gets an `error` frame and close code `4401`.
- **Size limit:** frames above `ws_max_message_bytes` (2 MB) are rejected by the server.
- **Transport:** ordered and reliable while open. Reconnect is the client's job.

## Client → server frames

### `hello` (required first frame)

```json
{
  "type": "hello",
  "session_id": "9f1c…",
  "protocol_version": 1,
  "client": { "name": "cuepilot-desktop", "platform": "darwin" }
}
```

| Field | Type | Notes |
|---|---|---|
| `session_id` | string \| null | Omit on first connect. Send a stored id to resume. |
| `protocol_version` | int | Defaults to `1`; must match. |
| `client` | object | Free-form metadata stored on the session (truncated). |

The server must receive `hello` within `ws_hello_timeout_s` (10 s) or it closes with
`hello_timeout`. Any other first frame yields `hello_required`.

### `prompt`

```json
{
  "type": "prompt",
  "id": "turn-42",
  "text": "What does CUE-E1042 mean?",
  "options": {
    "mode": "hybrid",
    "tools": true,
    "top_k": 8,
    "temperature": 0.2,
    "max_tokens": 700
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | string (1–128) | required | Turn id; echoed on every frame for this turn. |
| `text` | string (1–8000) | required | The user prompt. |
| `options.mode` | `hybrid` \| `bm25` \| `dense` | `hybrid` | Retrieval mode. |
| `options.tools` | bool | `true` | Advertise tools to the model. |
| `options.top_k` | int (1–20) | server default (8) | Fused result count. |
| `options.temperature` | float (0–2) | server default | Per-turn override. |
| `options.max_tokens` | int (16–8000) | server default | Per-turn override. |

Starting a prompt while another turn is active aborts the active turn (`superseded`) and
starts the new one. There is exactly one active turn per session.

### `abort`

```json
{ "type": "abort", "id": "turn-42" }
```

`id` is optional. Omitted → abort whatever turn is active. If `id` names a turn not local
to this replica, the server publishes `turn.abort` on the bus and replies
`aborted` with reason `not_local`; the owning replica cancels it.

### `audio`

```json
{ "type": "audio", "seq": 7, "pcm": "<base64 int16 LE>", "sample_rate": 16000 }
```

| Field | Type | Notes |
|---|---|---|
| `seq` | int ≥ 0 | Monotonic per connection; used for ordering/debug. |
| `pcm` | base64 string | Raw 16-bit little-endian mono samples. Max ~1.4 MB encoded. |
| `sample_rate` | int (8000–48000) | Defaults to 16000. |

### `ping` / `pong`

```json
{ "type": "ping", "ts": 1730000000000 }
{ "type": "pong", "ts": 1730000000000 }
```

`ts` is echoed back. Either side may send either frame. The server also sends `ping` every
`ws_heartbeat_interval_s` (20 s); the client answers with `pong`.

## Server → client frames

### `ready`

Sent once after a valid `hello` (and again if a `hello` arrives later on the same socket).

```json
{
  "type": "ready",
  "session_id": "9f1c…",
  "resumed": false,
  "protocol_version": 1,
  "kb_version": 3,
  "model": "openai/gpt-4o-mini",
  "server_time": 1730000000000
}
```

`resumed: true` means the supplied `session_id` existed. `model` is informational; the
mock provider reports `mock/echo-1`.

### `history`

Only after a resumed handshake, before any turn frames. Replaces the client's transcript.

```json
{
  "type": "history",
  "messages": [
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": "…" }
  ]
}
```

At most the last 20 messages. Roles are only `user` and `assistant`.

A turn restored from a streaming checkpoint is flagged rather than presented as finished:

```json
{
  "type": "history",
  "messages": [
    { "role": "user", "content": "explain the retry policy", "interrupted": true },
    { "role": "assistant", "content": "The client backs off", "interrupted": true }
  ]
}
```

`interrupted: true` means the server recovered a partial turn (a crash or a replica that
stopped checkpointing). The text is the last checkpoint, not necessarily the final answer.
Clients should render it as partial and may offer a resend.

### `ack`

```json
{ "type": "ack", "turn_id": "turn-42", "ts": 1730000000000 }
```

Emitted immediately after accepting a prompt, before retrieval starts.

### `retrieval`

```json
{
  "type": "retrieval",
  "turn_id": "turn-42",
  "query": "What does CUE-E1042 mean?",
  "mode": "hybrid",
  "kb_version": 3,
  "cache_hit": false,
  "degraded": [],
  "embedder": "hash-feature-v1",
  "timings_ms": { "bm25": 0.51, "dense": 0.93, "fusion": 0.01 },
  "candidates": { "bm25": 50, "dense": 50 },
  "hits": [
    {
      "id": "kb-errors#0-1a2b…",
      "doc_id": "kb-errors",
      "title": "Error codes",
      "score": 0.032786,
      "meta": { "tags": ["errors"], "chunk_index": 0 },
      "explain": {
        "ranks": { "bm25": 1, "dense": 2 },
        "contributions": { "bm25": 0.016393, "dense": 0.016129 },
        "raw_scores": { "bm25": 12.41, "dense": 0.83 }
      }
    }
  ]
}
```

`hits[].content` is omitted in turn frames to keep the payload small. `degraded` names any
retriever that failed (`"bm25"`, `"dense"`, or `"retrieval"` when the whole step failed).

### `delta`

```json
{ "type": "delta", "turn_id": "turn-42", "seq": 1, "text": "CUE-E1042 means ", "coalesced": 3 }
```

| Field | Notes |
|---|---|
| `seq` | Monotonic per turn starting at 1. |
| `text` | One or more tokens concatenated. |
| `coalesced` | Number of provider tokens merged into this frame. |

Frames are coalesced to roughly one per 16 ms. The first frame of a turn is flushed
immediately. Text is ordered and lossless under normal backpressure.

### `tool_call_start`

```json
{
  "type": "tool_call_start",
  "turn_id": "turn-42",
  "call_id": "call_0",
  "name": "kb_search",
  "arguments": { "query": "CUE-E1042", "top_k": 4 }
}
```

### `tool_call_result`

```json
{
  "type": "tool_call_result",
  "turn_id": "turn-42",
  "call_id": "call_0",
  "name": "kb_search",
  "ok": true,
  "result": {
    "ok": true,
    "tool": "kb_search",
    "query": "CUE-E1042",
    "hits": [ { "id": "…", "doc_id": "kb-errors", "title": "…", "snippet": "…", "score": 0.03 } ],
    "kb_version": 3,
    "cache_hit": false,
    "degraded": [],
    "latency_ms": 1.8
  },
  "latency_ms": 1.8
}
```

`ok: false` carries `result.error` instead. Tools: `kb_search`, `kb_open`, `now`,
`kb_stats`. Unknown tools and invalid arguments are reported to the model, not raised.

### `done`

```json
{
  "type": "done",
  "turn_id": "turn-42",
  "finish_reason": "stop",
  "ttft_ms": 124.6,
  "total_ms": 1840.2,
  "usage": { "prompt_tokens": 812, "completion_tokens": 141, "total_tokens": 953 },
  "tool_iterations": 1,
  "model": "openai/gpt-4o-mini"
}
```

`finish_reason` is the provider's value, or `tool_iterations_exhausted`, or `error`.
`ttft_ms` is measured server-side from prompt decode to the first delta handed to the
socket writer; it is `null` if the provider produced no text.

### `aborted`

```json
{ "type": "aborted", "turn_id": "turn-42", "reason": "cancelled" }
```

Reasons: `cancelled` (abort), `superseded` (a new prompt arrived), `no_active_turn`,
`not_local` (abort for a turn owned by another replica).

### `error`

```json
{
  "type": "error",
  "code": "rate_limited",
  "message": "provider rate limited: …",
  "retryable": true,
  "retry_after_ms": 4000,
  "turn_id": "turn-42",
  "ts": 1730000000000
}
```

`turn_id` is present for turn-scoped errors; absent for connection/protocol errors.
`retry_after_ms` appears when the provider supplied `Retry-After`.

Error codes:

| Code | Scope | Meaning | Retryable |
|---|---|---|---|
| `bad_frame` | connection | Frame failed schema validation. Socket stays open. | no |
| `hello_required` | connection | First frame was not `hello`; socket closes. | no |
| `hello_timeout` | connection | No `hello` within 10 s; socket closes. | no |
| `protocol_mismatch` | connection | Unsupported `protocol_version`; socket closes. | no |
| `unauthorized` | connection | Bad/missing token; socket closes with 4401. | no |
| `client_rate_limited` | connection | Too many connection opens for this client in the window; socket closes with 4429. | yes |
| `connection_cap` | connection | This replica is at its concurrent-socket cap; socket closes with 4429. | yes |
| `rate_limited` | turn | Provider 429 after retries/fallback. | yes |
| `provider_unavailable` | turn | Provider 5xx/transport failure. | yes |
| `context_overflow` | turn | Prompt exceeded the model window. | no |
| `provider_stream_error` | turn | Error event mid-stream. | no |
| `internal_error` | turn | Unhandled server exception. | no |
| `audio_disabled` | connection | Audio ingest turned off. | no |
| `bad_audio_frame` | connection | Invalid base64 PCM. | no |

### `pong` / `ping`

```json
{ "type": "pong", "ts": 1730000000000, "server_time": 1730000000004 }
{ "type": "ping", "ts": 1730000000000 }
```

The client uses `pong.ts` to compute RTT.

### `audio_ack`

```json
{
  "type": "audio_ack",
  "seq": 7,
  "buffered_ms": 700,
  "rms": 0.0412,
  "transcribing": true
}
```

`rms` is normalized 0–1 over the buffered window (drives the level meter).
`transcribing` is true only when an OpenAI key is configured.

### `transcript`

```json
{ "type": "transcript", "text": "…", "final": true, "window_ms": 3000, "ts": 1730000000000 }
```

Best-effort rolling-window transcription; absence of transcripts is normal.

## Ordering guarantees

For a single turn:

```
ack → retrieval? → (delta | tool_call_start | tool_call_result)* → done | aborted | error
```

- `seq` is monotonic within a turn's deltas.
- The server flushes buffered text before enqueueing any control or terminal frame.
- The client applies pending text before handling `done`/`error`/`aborted`/`history`.
- Multiple tool calls in one iteration start together and may complete in any order; match
  by `call_id`.
- Frames for different turns do not interleave in practice because a session has one active
  turn, but clients should key state by `turn_id` regardless.

## Handshake and reconnect

```
client                                   server
  │ ── WS open ─────────────────────────▶ │
  │ ── hello {session_id?} ──────────────▶ │
  │ ◀──────────────── ready {session_id} ─ │
  │ ◀──────────────── history? ─────────── │   (only if resumed)
  │ ── prompt {id,text,options} ─────────▶ │
  │ ◀──────────────── ack ──────────────── │
  │ ◀──────────────── retrieval ────────── │
  │ ◀──────────────── delta* ───────────── │
  │ ◀──────────────── done ─────────────── │
```

Client reconnect policy (implemented in `desktop/src/useCueStream.ts`):

```
delay = max(250 ms, random() * min(8000, 250 * 2^attempt))
```

up to 8 attempts. The session id is persisted in `localStorage` under
`cuepilot.session_id`, so a reconnect resumes the conversation instead of starting over.
A successful `ready` resets the attempt counter.

## Close codes

| Code | Meaning |
|---|---|
| `1000` | Normal close (client shutdown / new session). |
| `1006` | Abnormal close (network, process death) — client reconnects. |
| `4400` | Protocol error: bad first frame, invalid JSON, version mismatch. |
| `4401` | Unauthorized: missing/wrong `?token=`. |
| `4408` | Stale: no client frame for 2.5× the heartbeat interval. |
| `4429` | Rate limited: too many opens in the window, or this replica is at its connection cap. |

## Minimal client example

```js
const ws = new WebSocket("ws://127.0.0.1:8000/ws/stream");
ws.onopen = () => ws.send(JSON.stringify({ type: "hello", protocol_version: 1 }));
ws.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  if (frame.type === "ready") {
    ws.send(JSON.stringify({
      type: "prompt",
      id: "turn-1",
      text: "What does CUE-E1042 mean?",
      options: { mode: "hybrid", tools: true },
    }));
  }
  if (frame.type === "delta") process.stdout.write(frame.text);
  if (frame.type === "done") console.log("\nttft", frame.ttft_ms, "ms");
};
```

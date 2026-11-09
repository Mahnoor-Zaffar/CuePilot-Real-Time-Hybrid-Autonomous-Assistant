"""Labelled queries for retrieval evaluation.

Three families, because the hybrid stack exists precisely because they fail
differently:

* ``identifier`` - exact codes, flags and API fields. BM25 should win; a dense
  model has no particular reason to place ``CUE-E1042`` near the passage that
  defines it.
* ``paraphrase`` - intent described in the user's own words, with no shared
  content words. A lexical retriever has nothing to match; a semantic one does.
* ``near_miss`` - identifier queries against a corpus that contains decoys one
  character away (``CUE-E1041`` beside ``CUE-E1042``). Tests precision, not just
  recall.

``expected`` lists every document that is a defensible answer, so recall is not
penalised for a different-but-correct chunk. Labels are written from the corpus
text, not from retriever output: each entry below can be checked by reading the
named document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["identifier", "paraphrase", "near_miss"]


@dataclass(frozen=True)
class Query:
    text: str
    expected: tuple[str, ...]
    family: Family


QUERIES: list[Query] = [
    # ── identifier: exact codes, flags and fields ──────────────────────────
    Query("CUE-E1042", ("kb-error-codes",), "identifier"),
    Query("CUE-E2041 retry_after_ms", ("kb-error-codes", "kb-rate-limits"), "identifier"),
    Query("CUE-E4021", ("kb-error-codes",), "identifier"),
    Query("CUE-E5032", ("kb-error-codes", "kb-tool-calling"), "identifier"),
    Query("CUE-E6020", ("kb-billing",), "identifier"),
    Query("CUE-E3010", ("kb-error-codes",), "identifier"),
    Query("retry_backoff_cap_ms", ("kb-websocket-reconnect",), "identifier"),
    Query("ws_send_queue_size", ("kb-backpressure",), "identifier"),
    Query("delta_flush_ms", ("kb-backpressure",), "identifier"),
    Query("tool_timeout_s", ("kb-tool-calling",), "identifier"),
    Query("max_tool_iterations", ("kb-tool-calling",), "identifier"),
    Query("hnsw.ef_search", ("kb-pgvector-operations",), "identifier"),
    Query("vector_cosine_ops", ("kb-pgvector-operations",), "identifier"),
    Query("kb_meta.version", ("kb-pgvector-operations",), "identifier"),
    Query("retrieval_candidate_k", ("kb-hybrid-retrieval",), "identifier"),
    Query("rrf_k", ("kb-hybrid-retrieval",), "identifier"),
    Query("turn_history_limit", ("kb-session-state", "kb-billing"), "identifier"),
    Query("session_ttl_s", ("kb-session-state",), "identifier"),
    Query("transcription_model whisper-1", ("kb-transcription",), "identifier"),
    Query("audio_window_ms", ("kb-audio-capture",), "identifier"),
    Query("CmdOrCtrl+Shift+Space", ("kb-electron-hotkeys",), "identifier"),
    Query(
        "requestAnimationFrame batching",
        ("kb-sdk-typescript", "kb-frontend-performance"),
        "identifier",
    ),
    Query("llm_fallback_models", ("kb-rate-limits",), "identifier"),
    Query("provider_fallbacks", ("kb-rate-limits",), "identifier"),
    Query("ws_max_message_bytes", ("kb-audio-capture",), "identifier"),
    Query("retrieval_top_k", ("kb-hybrid-retrieval", "kb-billing"), "identifier"),
    Query("cue:context IPC", ("kb-electron-hotkeys",), "identifier"),
    Query("/api/traces/{trace_id}", ("kb-telemetry",), "identifier"),
    Query("allow_degraded_infra", ("kb-pgvector-operations", "kb-deployment"), "identifier"),
    Query("kb_search tool", ("kb-tool-calling", "kb-hybrid-retrieval"), "identifier"),
    # ── paraphrase: intent, no shared content words ────────────────────────
    Query("what order do messages arrive in during one answer", ("kb-streaming-protocol",), "paraphrase"),
    Query("how does the client know an answer is finished", ("kb-streaming-protocol",), "paraphrase"),
    Query("which message types can the browser send to the server", ("kb-streaming-protocol",), "paraphrase"),
    Query("how do I stop every desktop client reconnecting at the same time", ("kb-websocket-reconnect",), "paraphrase"),
    Query("what happens when the connection silently dies", ("kb-websocket-reconnect",), "paraphrase"),
    Query("how long does the client wait between reconnection attempts", ("kb-websocket-reconnect",), "paraphrase"),
    Query("why would semantic search return nothing at all", ("kb-error-codes", "kb-hybrid-retrieval"), "paraphrase"),
    Query("what does it mean when the model input is too long", ("kb-error-codes",), "paraphrase"),
    Query("how are keyword and vector results combined", ("kb-hybrid-retrieval",), "paraphrase"),
    Query("why not use a cross encoder to rerank the candidates", ("kb-hybrid-retrieval",), "paraphrase"),
    Query("how long are search results cached for", ("kb-hybrid-retrieval",), "paraphrase"),
    Query("how do I make the vector index work harder for better recall", ("kb-pgvector-operations",), "paraphrase"),
    Query("what should I run after a large re-ingest", ("kb-pgvector-operations",), "paraphrase"),
    Query("what happens to search if postgres is down", ("kb-pgvector-operations", "kb-deployment"), "paraphrase"),
    Query("how much conversation history goes into the prompt", ("kb-session-state",), "paraphrase"),
    Query("where is conversation state kept so a reconnect works", ("kb-session-state",), "paraphrase"),
    Query("how often is session state written to redis", ("kb-session-state",), "paraphrase"),
    Query("how do I cancel an answer that is already streaming", ("kb-cancellation",), "paraphrase"),
    Query("what happens if I send a new question while one is running", ("kb-cancellation",), "paraphrase"),
    Query("does cancelling stop the provider charging me", ("kb-cancellation", "kb-billing"), "paraphrase"),
    Query("what happens when the model provider says too many requests", ("kb-rate-limits",), "paraphrase"),
    Query("when does the gateway switch to a different model", ("kb-rate-limits",), "paraphrase"),
    Query("why won't it retry in the middle of a response", ("kb-rate-limits",), "paraphrase"),
    Query("how many rounds of tool use are allowed", ("kb-tool-calling",), "paraphrase"),
    Query("what happens when a tool takes too long", ("kb-tool-calling",), "paraphrase"),
    Query("why expose search as a tool instead of doing it up front", ("kb-tool-calling",), "paraphrase"),
    Query("what does the desktop app read from the clipboard", ("kb-electron-hotkeys",), "paraphrase"),
    Query("how does the renderer avoid getting node access", ("kb-electron-hotkeys", "kb-security"), "paraphrase"),
    Query("how are microphone recordings turned into text", ("kb-transcription", "kb-audio-capture"), "paraphrase"),
    Query("what audio format is sent over the socket", ("kb-audio-capture",), "paraphrase"),
    Query("how large is the rolling audio buffer", ("kb-audio-capture",), "paraphrase"),
    Query("how are partial captions produced", ("kb-transcription",), "paraphrase"),
    Query("how far behind speech do the captions lag", ("kb-transcription",), "paraphrase"),
    Query("where can I inspect the trace for a single turn", ("kb-telemetry",), "paraphrase"),
    Query("when does the first token timer start", ("kb-telemetry",), "paraphrase"),
    Query("which spans are recorded for a turn", ("kb-telemetry",), "paraphrase"),
    Query("how do I run the gateway across several servers", ("kb-deployment",), "paraphrase"),
    Query("do I need sticky sessions", ("kb-deployment",), "paraphrase"),
    Query("what does a replica do if redis disappears", ("kb-deployment",), "paraphrase"),
    Query("protecting the assistant from instructions hidden inside documents", ("kb-security",), "paraphrase"),
    Query("how do clients authenticate to the socket", ("kb-security",), "paraphrase"),
    Query("can retrieved text trigger a tool call", ("kb-security",), "paraphrase"),
    Query("what gets billed when a stream is cancelled halfway", ("kb-billing",), "paraphrase"),
    Query("what happens when a workspace runs out of budget", ("kb-billing",), "paraphrase"),
    Query("what actually drives the cost of a turn", ("kb-billing",), "paraphrase"),
    Query("python library usage example", ("kb-sdk-python",), "paraphrase"),
    Query("how does the python sdk handle reconnects", ("kb-sdk-python",), "paraphrase"),
    Query("react hook that batches incoming tokens", ("kb-sdk-typescript",), "paraphrase"),
    Query("can the typescript client run outside electron", ("kb-sdk-typescript",), "paraphrase"),
    Query("how much memory should the desktop app use while sitting idle", ("kb-frontend-performance",), "paraphrase"),
    Query("why is markdown parsing expensive while streaming", ("kb-frontend-performance",), "paraphrase"),
    Query("how does the server avoid flooding a slow client with tokens", ("kb-backpressure",), "paraphrase"),
    Query("what happens to tokens when the send queue is full", ("kb-backpressure",), "paraphrase"),
    Query("how many frames per second does the server emit", ("kb-backpressure",), "paraphrase"),
    # ── near_miss: decoys one character away ───────────────────────────────
    Query("CUE-E1042 context window exceeded", ("kb-error-codes",), "near_miss"),
    Query("CUE-E2041 provider rate limit", ("kb-error-codes", "kb-rate-limits"), "near_miss"),
    Query("CUE-E5032 tool timeout", ("kb-error-codes", "kb-tool-calling"), "near_miss"),
    Query("CUE-E6020 workspace budget", ("kb-billing",), "near_miss"),
    Query("retry_backoff_cap_ms reconnect delay", ("kb-websocket-reconnect",), "near_miss"),
    Query("tool_timeout_s tool dispatch", ("kb-tool-calling",), "near_miss"),
    Query("delta_flush_ms coalescing window", ("kb-backpressure",), "near_miss"),
    Query("ws_send_queue_size send queue", ("kb-backpressure",), "near_miss"),
    Query("hnsw.ef_search recall tradeoff", ("kb-pgvector-operations",), "near_miss"),
    Query("vector_cosine_ops index operator class", ("kb-pgvector-operations",), "near_miss"),
    Query("turn_history_limit prompt messages", ("kb-session-state", "kb-billing"), "near_miss"),
    Query("whisper-1 transcription model", ("kb-transcription",), "near_miss"),
    Query("CmdOrCtrl+Shift+Space global shortcut", ("kb-electron-hotkeys",), "near_miss"),
    Query("max_tool_iterations tool rounds", ("kb-tool-calling",), "near_miss"),
    Query("requestAnimationFrame delta batching", ("kb-sdk-typescript", "kb-frontend-performance"), "near_miss"),
    Query("kb_meta.version cache invalidation", ("kb-pgvector-operations",), "near_miss"),
]


def by_family(family: str) -> list[Query]:
    return [q for q in QUERIES if q.family == family]


def families() -> list[str]:
    seen: list[str] = []
    for query in QUERIES:
        if query.family not in seen:
            seen.append(query.family)
    return seen

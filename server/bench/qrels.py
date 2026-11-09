"""Labelled queries for retrieval evaluation.

Two query families, because the hybrid stack exists precisely because they fail
differently:
  * `identifier` - exact codes, flags and API fields. BM25 should win.
  * `paraphrase` - intent described in the user's own words. Dense should win.

`expected` lists every doc that is a defensible answer, so recall is not penalised
for a different-but-correct chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Query:
    text: str
    expected: tuple[str, ...]
    family: Literal["identifier", "paraphrase"]


QUERIES: list[Query] = [
    # ── identifier family ──────────────────────────────────────────────────
    Query("CUE-E1042", ("kb-error-codes",), "identifier"),
    Query("CUE-E2041 retry_after_ms", ("kb-error-codes", "kb-rate-limits"), "identifier"),
    Query("CUE-E4021", ("kb-error-codes",), "identifier"),
    Query("retry_backoff_cap_ms", ("kb-websocket-reconnect",), "identifier"),
    Query("ws_send_queue_size", ("kb-backpressure",), "identifier"),
    Query("tool_timeout_s", ("kb-tool-calling",), "identifier"),
    Query("hnsw.ef_search", ("kb-pgvector-operations",), "identifier"),
    Query("vector_cosine_ops", ("kb-pgvector-operations",), "identifier"),
    Query("transcription_model whisper-1", ("kb-transcription",), "identifier"),
    Query("CmdOrCtrl+Shift+Space", ("kb-electron-hotkeys",), "identifier"),
    Query("requestAnimationFrame batching", ("kb-sdk-typescript", "kb-frontend-performance"), "identifier"),
    Query("turn_history_limit", ("kb-session-state", "kb-billing"), "identifier"),
    Query("llm_fallback_models", ("kb-rate-limits",), "identifier"),
    Query("kb_meta.version", ("kb-pgvector-operations",), "identifier"),
    # ── paraphrase family ──────────────────────────────────────────────────
    Query("how do I stop every desktop client reconnecting at the same time", ("kb-websocket-reconnect",), "paraphrase"),
    Query("what happens when the model provider says too many requests", ("kb-rate-limits",), "paraphrase"),
    Query("how are microphone recordings turned into text", ("kb-transcription", "kb-audio-capture"), "paraphrase"),
    Query("how do I cancel an answer that is already streaming", ("kb-cancellation",), "paraphrase"),
    Query("why would semantic search return nothing at all", ("kb-error-codes", "kb-hybrid-retrieval"), "paraphrase"),
    Query("how much memory should the desktop app use while sitting idle", ("kb-frontend-performance",), "paraphrase"),
    Query("how does the server avoid flooding a slow client with tokens", ("kb-backpressure",), "paraphrase"),
    Query("how do I run the gateway across several servers", ("kb-deployment",), "paraphrase"),
    Query("protecting the assistant from instructions hidden inside documents", ("kb-security",), "paraphrase"),
    Query("what gets billed when a stream is cancelled halfway", ("kb-billing",), "paraphrase"),
    Query("python library usage example", ("kb-sdk-python",), "paraphrase"),
    Query("react hook that batches incoming tokens", ("kb-sdk-typescript",), "paraphrase"),
    Query("where can I inspect the trace for a single turn", ("kb-telemetry",), "paraphrase"),
    Query("what does the desktop app read from the clipboard", ("kb-electron-hotkeys",), "paraphrase"),
    Query("splitting camelCase and snake_case identifiers for search", ("kb-hybrid-retrieval",), "paraphrase"),
    Query("exponential backoff with jitter on reconnect", ("kb-websocket-reconnect",), "paraphrase"),
]


def by_family(family: str) -> list[Query]:
    return [q for q in QUERIES if q.family == family]

"""Tool registry.

The model can only call tools declared here, and every argument is validated
against a hand-written schema before dispatch. Retrieved text can suggest a tool
call; it can never execute one.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.llm.base import ToolCall
from app.retrieval.orchestrator import HybridRetriever
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder


class ToolValidationError(ValueError):
    pass


@dataclass
class ToolContext:
    session_id: str
    turn_id: str
    scratchpad: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    name: str = ""

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "tool": self.name}
        if self.ok:
            payload.update(self.data if isinstance(self.data, dict) else {"result": self.data})
        else:
            payload["error"] = self.error
        payload["latency_ms"] = round(self.latency_ms, 2)
        return payload

    def as_tool_message(self) -> str:
        return json.dumps(self.to_wire(), separators=(",", ":"))


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _require_str(args: dict[str, Any], key: str, *, max_len: int = 600) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"`{key}` must be a non-empty string")
    if len(value) > max_len:
        raise ToolValidationError(f"`{key}` exceeds {max_len} characters")
    return value.strip()


def _optional_int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"`{key}` must be an integer")
    return max(lo, min(hi, value))


class ToolRegistry:
    def __init__(
        self,
        retriever: HybridRetriever,
        settings: Settings,
        metrics: Metrics,
        recorder: TraceRecorder,
    ) -> None:
        self.retriever = retriever
        self.settings = settings
        self.metrics = metrics
        self.recorder = recorder
        self._specs: dict[str, ToolSpec] = {
            spec.name: spec
            for spec in (
                self._kb_search_spec(),
                self._kb_open_spec(),
                self._now_spec(),
                self._kb_stats_spec(),
            )
        }

    # ── schemas ────────────────────────────────────────────────────────────
    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._specs.values()]

    def _kb_search_spec(self) -> ToolSpec:
        return ToolSpec(
            name="kb_search",
            description=(
                "Hybrid (BM25 + dense) search over the knowledge base. Use a second, "
                "narrower query if the first results miss. Exact identifiers and error "
                "codes work well."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 4},
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "bm25", "dense"],
                        "default": "hybrid",
                    },
                },
                "required": ["query"],
            },
            handler=self._kb_search,
        )

    def _kb_open_spec(self) -> ToolSpec:
        return ToolSpec(
            name="kb_open",
            description="Return the full text of a chunk id previously returned by kb_search.",
            parameters={
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
            handler=self._kb_open,
        )

    def _now_spec(self) -> ToolSpec:
        return ToolSpec(
            name="now",
            description="Current server time in ISO-8601 UTC. Use for time-relative questions.",
            parameters={"type": "object", "properties": {}},
            handler=self._now,
        )

    def _kb_stats_spec(self) -> ToolSpec:
        return ToolSpec(
            name="kb_stats",
            description="Knowledge base size, version and active embedder.",
            parameters={"type": "object", "properties": {}},
            handler=self._kb_stats,
        )

    # ── handlers ───────────────────────────────────────────────────────────
    async def _kb_search(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        query = _require_str(args, "query", max_len=500)
        top_k = _optional_int(args, "top_k", 4, 1, 12)
        mode = args.get("mode", "hybrid")
        if mode not in ("hybrid", "bm25", "dense"):
            mode = "hybrid"
        result = await self.retriever.search(query, top_k=top_k, mode=mode)
        hits = [
            {
                "id": hit.chunk.id,
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "snippet": " ".join(hit.chunk.content.split())[:400],
                "score": round(hit.score, 6),
                "ranks": hit.ranks,
            }
            for hit in result.hits
        ]
        return {
            "query": query,
            "hits": hits,
            "kb_version": result.kb_version,
            "cache_hit": result.cache_hit,
            "degraded": result.degraded,
        }

    async def _kb_open(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        chunk_id = _require_str(args, "chunk_id", max_len=200)
        chunk = await self.retriever.get_chunk(chunk_id)
        if chunk is None:
            raise ToolValidationError(f"unknown chunk_id `{chunk_id}`")
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "content": chunk.content[:4000],
        }

    async def _now(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {"iso": now.isoformat(), "epoch_ms": int(now.timestamp() * 1000)}

    async def _kb_stats(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        return await self.retriever.stats()

    # ── dispatch ───────────────────────────────────────────────────────────
    async def dispatch(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        spec = self._specs.get(call.name)
        with self.recorder.span(
            "tool.call", tool=call.name, call_id=call.id, turn_id=ctx.turn_id
        ) as span:
            if spec is None:
                result = ToolResult(ok=False, error=f"unknown tool `{call.name}`", name=call.name)
            else:
                try:
                    data = await asyncio.wait_for(
                        spec.handler(call.arguments, ctx), timeout=self.settings.tool_timeout_s
                    )
                    result = ToolResult(ok=True, data=data, name=call.name)
                except TimeoutError:
                    result = ToolResult(
                        ok=False,
                        error=f"tool timed out after {self.settings.tool_timeout_s}s",
                        name=call.name,
                    )
                except (ToolValidationError, ValueError) as exc:
                    result = ToolResult(ok=False, error=str(exc), name=call.name)
                except Exception as exc:  # noqa: BLE001 - surfaced to the model
                    result = ToolResult(
                        ok=False, error=f"{type(exc).__name__}: {exc}", name=call.name
                    )
            result.latency_ms = (time.perf_counter() - started) * 1000
            span.set(ok=result.ok, latency_ms=round(result.latency_ms, 2))
            self.metrics.observe("tool_call_ms", result.latency_ms)
            self.metrics.incr("tool_calls_total")
            if not result.ok:
                self.metrics.incr("tool_calls_failed")
            return result

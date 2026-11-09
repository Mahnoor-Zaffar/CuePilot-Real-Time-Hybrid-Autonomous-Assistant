from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from app.infra.cache import MemoryCache
from app.infra.db import MemoryVectorStore
from app.llm.agent import Agent, TurnRequest
from app.llm.base import ProviderStreamError, RateLimitError, StreamEvent
from app.llm.tools import ToolRegistry
from app.retrieval.corpus import all_documents
from app.retrieval.embedder import HashEmbedder
from app.retrieval.orchestrator import HybridRetriever
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder
from tests.conftest import ScriptedProvider, make_settings, text_turn, tool_call_turn


@dataclass
class FakeSession:
    id: str = "sess-1"
    history: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)


async def build_agent(provider) -> tuple[Agent, FakeSession]:
    settings = make_settings()
    metrics = Metrics()
    recorder = TraceRecorder()
    retriever = HybridRetriever(
        settings,
        MemoryVectorStore(settings.embedding_dim),
        HashEmbedder(settings.embedding_dim),
        MemoryCache(),
        metrics,
        recorder,
    )
    await retriever.ingest(all_documents(include_distractors=False))
    tools = ToolRegistry(retriever, settings, metrics, recorder)
    return Agent(provider, retriever, tools, settings, metrics, recorder), FakeSession()


async def collect(agent: Agent, session: FakeSession, prompt: str = "What is CUE-E1042?") -> list[dict]:
    return [frame async for frame in agent.run(TurnRequest(turn_id="t1", prompt=prompt), session)]


async def test_tool_loop_runs_search_then_answers() -> None:
    provider = ScriptedProvider([tool_call_turn(), text_turn()])
    agent, session = await build_agent(provider)
    frames = await collect(agent, session)

    types = [frame["type"] for frame in frames]
    assert types[0] == "retrieval"
    assert "tool_call_start" in types
    assert "tool_call_result" in types
    assert types[-1] == "done"

    start = next(frame for frame in frames if frame["type"] == "tool_call_start")
    assert start["name"] == "kb_search"
    result = next(frame for frame in frames if frame["type"] == "tool_call_result")
    assert result["ok"] is True
    assert result["result"]["hits"]

    # second model call must see the tool message
    assert provider.received[-1][-1].role == "tool"
    done = frames[-1]
    assert done["ttft_ms"] is not None
    assert done["tool_iterations"] == 1
    assert done["usage"]["completion_tokens"] == 8


async def test_history_is_appended() -> None:
    provider = ScriptedProvider([text_turn()])
    agent, session = await build_agent(provider)
    await collect(agent, session, "hello there")
    assert session.history[0] == {"role": "user", "content": "hello there"}
    assert session.history[-1]["role"] == "assistant"
    assert "context window" in session.history[-1]["content"]


async def test_rate_limit_becomes_retryable_error_frame() -> None:
    class RateLimited(ScriptedProvider):
        async def stream(self, *args, **kwargs):
            raise RateLimitError("slow down", retry_after_ms=1500)
            yield  # pragma: no cover

    agent, session = await build_agent(RateLimited([]))
    frames = await collect(agent, session)
    types = [frame["type"] for frame in frames]
    assert "error" in types
    error = next(frame for frame in frames if frame["type"] == "error")
    assert error["code"] == "rate_limited"
    assert error["retryable"] is True
    assert error["retry_after_ms"] == 1500
    assert types[-1] == "done"
    assert frames[-1]["finish_reason"] == "error"


async def test_mid_stream_failure_is_not_retried() -> None:
    class MidStream(ScriptedProvider):
        async def stream(self, *args, **kwargs):
            yield StreamEvent(kind="delta", text="partial ")
            raise ProviderStreamError("upstream reset")

    agent, session = await build_agent(MidStream([]))
    frames = await collect(agent, session)
    assert [f["type"] for f in frames if f["type"] == "delta"] == ["delta"]
    assert any(f["type"] == "error" for f in frames)


async def test_cancellation_propagates() -> None:
    class Slow(ScriptedProvider):
        async def stream(self, *args, **kwargs):
            for _ in range(100):
                await asyncio.sleep(0.05)
                yield StreamEvent(kind="delta", text="tick ")

    agent, session = await build_agent(Slow([]))

    async def consume() -> None:
        async for _ in agent.run(TurnRequest(turn_id="t1", prompt="long"), session):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.08)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # partial assistant text is preserved for the transcript
    assert session.history[-1]["role"] == "assistant"


async def test_unknown_tool_is_reported_not_raised() -> None:
    provider = ScriptedProvider([tool_call_turn(name="rm_rf", arguments={}), text_turn("done")])
    agent, session = await build_agent(provider)
    frames = await collect(agent, session)
    result = next(frame for frame in frames if frame["type"] == "tool_call_result")
    assert result["ok"] is False
    assert "unknown tool" in result["result"]["error"]

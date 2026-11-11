from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from app.config import Settings
from app.llm.base import ChatMessage, ChatProvider, StreamEvent, ToolCall
from app.main import create_app
from fastapi.testclient import TestClient


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "test",
        "llm_provider": "mock",
        "embedding_provider": "hash",
        "allow_degraded_infra": True,
        "infra_probe_timeout_s": 0.05,
        "database_url": "postgresql://cuepilot:cuepilot@127.0.0.1:1/none",
        "redis_url": "redis://127.0.0.1:1/0",
        "otel_enabled": False,
        "otel_exporter": "none",
        "seed_on_start": True,
        "mock_token_delay_ms": 0.0,
        "mock_ttft_ms": 0.0,
        "audio_enabled": True,
        "api_token": None,
        # Tests must not leave spool files behind; spool tests opt in with a
        # tmp_path-backed path.
        "session_journal_spool_path": "",
        "metrics_publish_interval_s": 3600.0,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings() -> Settings:
    return make_settings()


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def runtime(client: TestClient):
    return client.app.state.runtime


def drain_ws(ws, *, until: str = "done", limit: int = 400) -> list[dict]:
    frames: list[dict] = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if frame.get("type") == until:
            return frames
    raise AssertionError(f"never received {until!r}; frames={[f.get('type') for f in frames]}")


class ScriptedProvider(ChatProvider):
    """Deterministic provider used to test the tool loop and cancellation."""

    name = "scripted"
    model = "scripted/test"

    def __init__(self, turns: list[list[StreamEvent]], delay_s: float = 0.0) -> None:
        self._turns = turns
        self._index = 0
        self.delay_s = delay_s
        self.received: list[list[ChatMessage]] = []

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> AsyncIterator[StreamEvent]:
        self.received.append(list(messages))
        turn = self._turns[min(self._index, len(self._turns) - 1)]
        self._index += 1
        for event in turn:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield event

    async def warm(self) -> None:
        return None

    async def close(self) -> None:
        return None


def tool_call_turn(name: str = "kb_search", arguments: dict | None = None) -> list[StreamEvent]:
    return [
        StreamEvent(
            kind="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name=name, arguments=arguments or {"query": "CUE-E1042"})
            ],
        ),
        StreamEvent(kind="finish", finish_reason="tool_calls"),
    ]


def text_turn(text: str = "The error means the context window overflowed.") -> list[StreamEvent]:
    words = text.split(" ")
    events = [
        StreamEvent(kind="delta", text=word if index == 0 else f" {word}")
        for index, word in enumerate(words)
    ]
    events.append(StreamEvent(kind="usage", usage={"prompt_tokens": 10, "completion_tokens": 8,
                                                   "total_tokens": 18}))
    events.append(StreamEvent(kind="finish", finish_reason="stop"))
    return events


__all__ = [
    "ScriptedProvider",
    "client",
    "drain_ws",
    "make_settings",
    "runtime",
    "settings",
    "text_turn",
    "tool_call_turn",
]

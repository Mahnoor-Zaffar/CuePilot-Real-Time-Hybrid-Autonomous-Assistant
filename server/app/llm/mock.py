"""Deterministic mock provider.

Purpose: the full pipeline (retrieval -> tool call -> tool result -> streamed
answer) must run with no API key, in CI, and in benchmarks where we need a stable
TTFT baseline. It is never presented as a real model.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.llm.base import ChatMessage, ChatProvider, StreamEvent, ToolCall

_CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.S)


class MockProvider(ChatProvider):
    name = "mock"
    model = "mock/echo-1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def warm(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(self.settings.mock_ttft_ms / 1000)

        user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"), "the request"
        )
        tool_messages = [m for m in messages if m.role == "tool"]
        assistant_so_far = "".join(m.content for m in messages if m.role == "assistant")

        # First pass with tools available: exercise the tool loop.
        if tools and not tool_messages and not assistant_so_far:
            yield StreamEvent(
                kind="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="mock_call_1",
                        name="kb_search",
                        arguments={"query": user_text[:300], "top_k": 4},
                    )
                ],
            )
            yield StreamEvent(kind="finish", finish_reason="tool_calls")
            return

        answer = self._compose(user_text, messages, tool_messages)
        words = answer.split(" ")
        for index, word in enumerate(words):
            chunk = word if index == 0 else f" {word}"
            yield StreamEvent(kind="delta", text=chunk)
            if self.settings.mock_token_delay_ms:
                await asyncio.sleep(self.settings.mock_token_delay_ms / 1000)
        completion = len(words)
        yield StreamEvent(
            kind="usage",
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in messages),
                "completion_tokens": completion,
                "total_tokens": completion,
            },
        )
        yield StreamEvent(kind="finish", finish_reason="stop")

    def _compose(
        self, user_text: str, messages: list[ChatMessage], tool_messages: list[ChatMessage]
    ) -> str:
        sources: list[str] = []
        for message in tool_messages:
            try:
                payload = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for hit in payload.get("hits", [])[:2]:
                title = hit.get("title") or hit.get("doc_id", "source")
                snippet = " ".join(str(hit.get("snippet", "")).split())
                sources.append(f"{title}: {snippet[:220]}")

        if not sources:
            system = next((m.content for m in messages if m.role == "system"), "")
            match = _CONTEXT_RE.search(system)
            if match:
                block = " ".join(match.group(1).split())
                sources.append(block[:260])

        if sources:
            body = " ".join(sources[:2])
            return (
                f"Based on the indexed knowledge base: {body} "
                f"(mock provider: no external model was called for this answer.)"
            )
        return (
            f"I could not find indexed context for '{user_text[:120]}'. "
            "Try ingesting documents with POST /api/kb/ingest first. "
            "(mock provider: no external model was called.)"
        )

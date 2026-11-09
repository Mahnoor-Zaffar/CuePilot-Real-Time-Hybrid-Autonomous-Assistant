"""Provider-agnostic streaming chat types.

The gateway only ever consumes `StreamEvent`s, so swapping OpenRouter for a local
vLLM server or the deterministic mock is a registry change, not an agent change.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class StreamEvent:
    kind: Literal["delta", "tool_calls", "usage", "finish"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class ProviderError(Exception):
    """Base class for provider failures that the agent can classify."""

    code = "provider_error"
    retryable = False

    def __init__(self, message: str, *, retry_after_ms: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms


class RateLimitError(ProviderError):
    code = "rate_limited"
    retryable = True


class ProviderUnavailable(ProviderError):
    code = "provider_unavailable"
    retryable = True


class ProviderStreamError(ProviderError):
    """Failure after tokens were emitted; retrying would corrupt the stream."""

    code = "provider_stream_error"
    retryable = False


class ContextOverflow(ProviderError):
    code = "context_overflow"
    retryable = False


class ChatProvider(Protocol):
    name: str
    model: str

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> AsyncIterator[StreamEvent]: ...

    async def warm(self) -> None: ...

    async def close(self) -> None: ...

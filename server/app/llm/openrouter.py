"""OpenRouter provider (OpenAI-compatible chat completions over SSE).

Handles the things that actually bite in production:
  * retries with exponential backoff + full jitter, honouring Retry-After;
  * never retries after the first token (that would duplicate text);
  * classifies 429 / 5xx / context-overflow into typed errors;
  * reassembles streamed tool-call fragments by index;
  * requests usage on the final chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.llm.base import (
    ChatMessage,
    ChatProvider,
    ContextOverflow,
    ProviderError,
    ProviderStreamError,
    ProviderUnavailable,
    RateLimitError,
    StreamEvent,
    ToolCall,
)
from app.telemetry.metrics import Metrics

logger = logging.getLogger(__name__)


def _retry_after_ms(response: httpx.Response) -> int | None:
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return int(float(header) * 1000)
    except ValueError:
        return None


class OpenRouterProvider(ChatProvider):
    name = "openrouter"

    def __init__(self, settings: Settings, model: str, metrics: Metrics | None = None) -> None:
        self.settings = settings
        self.model = model
        self.metrics = metrics
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=httpx.Timeout(
                settings.llm_timeout_s, connect=settings.llm_connect_timeout_s
            ),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": settings.openrouter_app_url,
                "X-Title": settings.openrouter_app_name,
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    # ── public API ─────────────────────────────────────────────────────────
    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        attempt = 0
        while True:
            emitted = False
            try:
                async for event in self._attempt(payload):
                    emitted = True
                    yield event
                return
            except ProviderError as exc:
                if emitted or not exc.retryable or attempt >= self.settings.llm_max_retries:
                    raise
                attempt += 1
                delay = self._backoff(attempt, exc.retry_after_ms)
                if self.metrics is not None:
                    self.metrics.incr("provider_retries")
                logger.warning(
                    "provider retry %s/%s model=%s code=%s in %.2fs",
                    attempt,
                    self.settings.llm_max_retries,
                    self.model,
                    exc.code,
                    delay,
                )
                await asyncio.sleep(delay)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                if emitted or attempt >= self.settings.llm_max_retries:
                    raise ProviderUnavailable(f"transport failure: {exc}") from exc
                attempt += 1
                await asyncio.sleep(self._backoff(attempt, None))

    async def warm(self) -> None:
        """No-op: HTTP keep-alive pool is created eagerly in __init__."""

    async def close(self) -> None:
        await self._client.aclose()

    # ── internals ──────────────────────────────────────────────────────────
    def _backoff(self, attempt: int, retry_after_ms: int | None) -> float:
        if retry_after_ms is not None:
            return min(retry_after_ms / 1000, 10.0)
        base = self.settings.llm_retry_base_s * (2 ** (attempt - 1))
        return random.uniform(0, min(base, 8.0))  # full jitter

    async def _attempt(self, payload: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] | None = None
        finish_reason: str | None = None

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:600]
                raise self._classify(response, body)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    raise ProviderStreamError(str(chunk["error"])[:400])

                if chunk.get("usage"):
                    raw = chunk["usage"]
                    usage = {
                        "prompt_tokens": int(raw.get("prompt_tokens", 0)),
                        "completion_tokens": int(raw.get("completion_tokens", 0)),
                        "total_tokens": int(raw.get("total_tokens", 0)),
                    }

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    text = delta.get("content")
                    if text:
                        yield StreamEvent(kind="delta", text=text)
                    for fragment in delta.get("tool_calls") or []:
                        index = int(fragment.get("index", 0))
                        slot = tool_calls.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if fragment.get("id"):
                            slot["id"] = fragment["id"]
                        function = fragment.get("function") or {}
                        if function.get("name"):
                            slot["name"] = function["name"]
                        if function.get("arguments"):
                            slot["arguments"] += function["arguments"]

        if tool_calls:
            parsed: list[ToolCall] = []
            for index in sorted(tool_calls):
                slot = tool_calls[index]
                try:
                    arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": slot["arguments"]}
                parsed.append(
                    ToolCall(
                        id=slot["id"] or f"call_{index}",
                        name=slot["name"] or "unknown",
                        arguments=arguments,
                    )
                )
            yield StreamEvent(kind="tool_calls", tool_calls=parsed)

        if usage:
            yield StreamEvent(kind="usage", usage=usage)
        yield StreamEvent(kind="finish", finish_reason=finish_reason or "stop")

    def _classify(self, response: httpx.Response, body: str) -> ProviderError:
        lowered = body.lower()
        if response.status_code == 429:
            return RateLimitError(
                f"provider rate limited: {body}", retry_after_ms=_retry_after_ms(response)
            )
        if response.status_code in (500, 502, 503, 504, 408):
            return ProviderUnavailable(f"provider {response.status_code}: {body}")
        if "context" in lowered and ("length" in lowered or "window" in lowered):
            return ContextOverflow(f"context overflow: {body}")
        return ProviderError(f"provider {response.status_code}: {body}")

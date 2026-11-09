"""Provider registry and fallback chain."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.llm.base import ChatMessage, ChatProvider, ProviderUnavailable, RateLimitError, StreamEvent
from app.llm.mock import MockProvider
from app.llm.openrouter import OpenRouterProvider
from app.telemetry.metrics import Metrics

logger = logging.getLogger(__name__)


class FallbackChain(ChatProvider):
    """Try each model in order, but only before the first token is emitted.

    Mid-stream switching would duplicate or reorder text, so it is never done.
    """

    name = "openrouter-chain"

    def __init__(self, providers: list[ChatProvider], metrics: Metrics) -> None:
        if not providers:
            raise ValueError("fallback chain requires at least one provider")
        self._providers = providers
        self.metrics = metrics

    @property
    def model(self) -> str:
        return self._providers[0].model

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> AsyncIterator[StreamEvent]:
        last_error: Exception | None = None
        for index, provider in enumerate(self._providers):
            emitted = False
            try:
                async for event in provider.stream(
                    messages, tools=tools, temperature=temperature, max_tokens=max_tokens
                ):
                    emitted = True
                    yield event
                return
            except (RateLimitError, ProviderUnavailable) as exc:
                last_error = exc
                if emitted or index == len(self._providers) - 1:
                    raise
                self.metrics.incr("provider_fallbacks")
                logger.warning(
                    "falling back from %s to %s after %s",
                    provider.model,
                    self._providers[index + 1].model,
                    exc.code,
                )
        if last_error is not None:
            raise last_error

    async def warm(self) -> None:
        for provider in self._providers:
            try:
                await provider.warm()
            except Exception:  # noqa: BLE001 - warming is best effort
                continue

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()


def build_provider(settings: Settings, metrics: Metrics) -> ChatProvider:
    if settings.provider_mode == "mock":
        return MockProvider(settings)
    providers: list[ChatProvider] = [
        OpenRouterProvider(settings, model, metrics=metrics) for model in settings.model_chain
    ]
    if len(providers) == 1:
        return providers[0]
    return FallbackChain(providers, metrics)

"""OpenTelemetry wiring plus a per-turn in-memory span buffer.

Two consumers, one span tree:
  1. a real OTel TracerProvider (console or OTLP/HTTP exporter) for external
     trace stores;
  2. a bounded ring buffer keyed by trace_id, exposed at
     `/api/traces/{trace_id}` so the portfolio demo can show socket
     lifecycle -> retrieval -> TTFT -> tool call timings without a collector.
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import Settings

logger = logging.getLogger(__name__)

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "cuepilot_span", default=None
)
_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cuepilot_trace_id", default=None
)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    started_at: float
    attributes: dict[str, Any] = field(default_factory=dict)
    ended_at: float | None = None
    status: str = "ok"
    error: str | None = None
    children: list[Span] = field(default_factory=list)
    _otel: Any = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)

    def set(self, **attrs: Any) -> None:
        self.attributes.update({k: v for k, v in attrs.items() if v is not None})
        if self._otel is not None:
            for key, value in attrs.items():
                if value is not None:
                    self._otel.set_attribute(key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


class TraceRecorder:
    """Bounded LRU of trace trees, newest first."""

    def __init__(self, max_traces: int = 200) -> None:
        self._traces: OrderedDict[str, Span] = OrderedDict()
        self._max = max_traces

    def _remember(self, span: Span) -> None:
        self._traces[span.trace_id] = self._traces.pop(span.trace_id, span)
        self._traces.move_to_end(span.trace_id)
        while len(self._traces) > self._max:
            self._traces.popitem(last=False)

    @contextmanager
    def trace(self, name: str, **attrs: Any) -> Iterator[Span]:
        trace_id = uuid.uuid4().hex
        span = Span(name=name, trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                    parent_id=None, started_at=time.perf_counter())
        span.set(**attrs)
        self._remember(span)
        token_span = _current_span.set(span)
        token_trace = _current_trace.set(trace_id)
        otel_span = self._otel_span(name, span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.ended_at = time.perf_counter()
            if otel_span is not None:
                if span.status == "error":
                    otel_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, span.error))
                otel_span.end()
            _current_span.reset(token_span)
            _current_trace.reset(token_trace)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current_span.get()
        trace_id = _current_trace.get() or uuid.uuid4().hex
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            started_at=time.perf_counter(),
        )
        span.set(**attrs)
        if parent is not None:
            parent.children.append(span)
        else:
            self._remember(span)
        token = _current_span.set(span)
        otel_span = self._otel_span(name, span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.ended_at = time.perf_counter()
            if otel_span is not None:
                if span.status == "error":
                    otel_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, span.error))
                otel_span.end()
            _current_span.reset(token)

    def _otel_span(self, name: str, span: Span) -> Any:
        tracer = otel_trace.get_tracer("cuepilot")
        try:
            otel_span = tracer.start_span(name)
            for key, value in span.attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    otel_span.set_attribute(key, value)
            return otel_span
        except Exception:  # pragma: no cover - tracing must never break a request
            return None

    def get(self, trace_id: str) -> dict[str, Any] | None:
        span = self._traces.get(trace_id)
        return span.to_dict() if span else None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        roots = list(self._traces.values())[-limit:][::-1]
        return [s.to_dict() for s in roots]


def current_trace_id() -> str | None:
    return _current_trace.get()


def configure_tracing(settings: Settings) -> TraceRecorder:
    """Idempotent tracer setup; returns the process-wide recorder."""
    recorder = TraceRecorder(max_traces=settings.trace_buffer_turns)
    if not settings.otel_enabled or settings.otel_exporter == "none":
        return recorder

    provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
    try:
        if settings.otel_exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint}/v1/traces")
        else:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as exc:  # pragma: no cover - exporter optional
        logger.warning("otel exporter unavailable: %s", exc)
    otel_trace.set_tracer_provider(provider)
    return recorder

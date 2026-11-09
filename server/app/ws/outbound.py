"""Bounded outbound writer with delta coalescing.

Server-side backpressure in three moves:
  1. token deltas never touch the socket directly - they land in a per-turn buffer;
  2. a flush window (`delta_flush_ms`, default 16ms) merges every buffered delta
     into one frame, capping outbound frame rate near 60/s regardless of token rate;
  3. the send queue is bounded. When it is full, delta *frames* are dropped but
     their text is re-queued for the next flush - so we shed frame pressure, never
     tokens. Control frames (done, tool_call_*, error) are never dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from fastapi import WebSocket

from app.config import Settings
from app.telemetry.metrics import Metrics
from app.ws import protocol

logger = logging.getLogger(__name__)

MAX_BUFFER_CHARS = 64_000


class ConnectionClosed(RuntimeError):
    pass


class OutboundStream:
    def __init__(self, websocket: WebSocket, settings: Settings, metrics: Metrics) -> None:
        self._ws = websocket
        self.settings = settings
        self.metrics = metrics
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=settings.ws_send_queue_size
        )
        self._buffers: dict[str, list[str]] = {}
        self._buffer_chars: dict[str, int] = {}
        self._token_counts: dict[str, int] = {}
        self._seq: dict[str, int] = {}
        self._flush_event = asyncio.Event()
        self._buffer_lock = asyncio.Lock()
        self._writer: asyncio.Task | None = None
        self._flusher: asyncio.Task | None = None
        self._closed = False
        self._closing = False
        self.error: BaseException | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._writer = asyncio.create_task(self._writer_loop(), name="ws-writer")
        self._flusher = asyncio.create_task(self._flush_loop(), name="ws-flusher")

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._closed = True
        with contextlib.suppress(Exception):
            await self._flush_all()
        self._flush_event.set()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)
        for task in (self._flusher, self._writer):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ── control frames (never dropped) ─────────────────────────────────────
    async def send(self, frame: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionClosed("outbound stream closed")
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Deltas occupy the queue; control frames jump the line by waiting.
            self.metrics.incr("control_frames_waited")
            try:
                await asyncio.wait_for(self._queue.put(frame), timeout=3.0)
            except TimeoutError as exc:
                self._closed = True
                raise ConnectionClosed("send queue wedged") from exc

    async def drain(self, timeout_s: float = 2.0) -> None:
        """Wait until every queued frame has been handed to the socket."""
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._queue.join(), timeout=timeout_s)

    # ── delta path ─────────────────────────────────────────────────────────
    async def push_delta(self, turn_id: str, text: str) -> None:
        if not text or self._closed:
            return
        async with self._buffer_lock:
            self._buffers.setdefault(turn_id, []).append(text)
            self._buffer_chars[turn_id] = self._buffer_chars.get(turn_id, 0) + len(text)
            self._token_counts[turn_id] = self._token_counts.get(turn_id, 0) + 1
            first = self._seq.get(turn_id, 0) == 0
        if first:
            # TTFT-sensitive path: the first token must not wait for a flush window.
            await self._flush_one(turn_id)
            return
        self._flush_event.set()

    async def flush_turn(self, turn_id: str) -> None:
        """Flush a single turn before its terminal frame, preserving order."""
        await self._flush_one(turn_id)

    # ── internals ──────────────────────────────────────────────────────────
    async def _flush_loop(self) -> None:
        window = self.settings.delta_flush_ms / 1000
        while not self._closed:
            try:
                await self._flush_event.wait()
                await asyncio.sleep(window)
                self._flush_event.clear()
                await self._flush_all()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("delta flush loop error")

    async def _flush_all(self) -> None:
        async with self._buffer_lock:
            turn_ids = [tid for tid, buf in self._buffers.items() if buf]
        for turn_id in turn_ids:
            await self._flush_one(turn_id)

    async def _flush_one(self, turn_id: str) -> None:
        async with self._buffer_lock:
            parts = self._buffers.get(turn_id) or []
            if not parts:
                return
            text = "".join(parts)
            coalesced = self._token_counts.pop(turn_id, len(parts))
            self._buffers[turn_id] = []
            self._buffer_chars[turn_id] = 0
            seq = self._seq.get(turn_id, 0) + 1
            self._seq[turn_id] = seq
        frame = protocol.delta(turn_id, seq, text, coalesced=coalesced)
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Shed the frame, keep the text: next flush sends it in one frame.
            self.metrics.incr("delta_frames_dropped")
            async with self._buffer_lock:
                pending = self._buffers.get(turn_id) or []
                pending.insert(0, text)
                self._buffers[turn_id] = pending
                self._buffer_chars[turn_id] = self._buffer_chars.get(turn_id, 0) + len(text)
                if self._buffer_chars[turn_id] > MAX_BUFFER_CHARS:
                    dropped = self._buffers[turn_id][0]
                    self._buffers[turn_id] = self._buffers[turn_id][1:]
                    self._buffer_chars[turn_id] -= len(dropped)
                    self.metrics.incr("delta_chars_dropped", len(dropped))
            self._flush_event.set()

    async def _writer_loop(self) -> None:
        while True:
            frame = await self._queue.get()
            if frame is None:
                self._queue.task_done()
                return
            started = time.perf_counter()
            try:
                await asyncio.wait_for(self._ws.send_json(frame), timeout=5.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - socket is gone
                self.error = exc
                self._closed = True
                self.metrics.incr("ws_send_failures")
                self._queue.task_done()
                return
            self._queue.task_done()
            self.metrics.observe("ws_send_ms", (time.perf_counter() - started) * 1000)
            self.metrics.incr("frames_sent")
            if frame.get("type") == "delta":
                self.metrics.incr("delta_frames_sent")
                self.metrics.incr("delta_tokens_sent", int(frame.get("coalesced", 1)))

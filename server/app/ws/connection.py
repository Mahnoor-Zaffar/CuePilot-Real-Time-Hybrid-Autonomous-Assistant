"""Per-connection handler: handshake, receive loop, turn lifecycle, heartbeat."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect

from app.llm.agent import TurnRequest
from app.ws import protocol
from app.ws.audio import AudioIngest
from app.ws.outbound import ConnectionClosed, OutboundStream
from app.ws.protocol import (
    AbortFrame,
    AudioFrame,
    FrameError,
    HelloFrame,
    PingFrame,
    PongFrame,
    PromptFrame,
    parse_client_frame,
)
from app.ws.session import Session, TurnHandle

if TYPE_CHECKING:
    from app.runtime import Runtime

logger = logging.getLogger(__name__)

CLOSE_UNAUTHORIZED = 4401
CLOSE_PROTOCOL_ERROR = 4400
CLOSE_STALE = 4408


class Connection:
    def __init__(self, websocket: WebSocket, runtime: Runtime) -> None:
        self.ws = websocket
        self.runtime = runtime
        self.settings = runtime.settings
        self.metrics = runtime.metrics
        self.recorder = runtime.recorder
        self.outbound = OutboundStream(websocket, self.settings, self.metrics)
        self.session: Session | None = None
        self.audio: AudioIngest | None = None
        self._turn_tasks: set[asyncio.Task] = set()
        self._heartbeat: asyncio.Task | None = None
        self._closing = False

    # ── entrypoint ─────────────────────────────────────────────────────────
    async def run(self) -> None:
        await self.ws.accept()
        self.metrics.incr("ws_connections_total")
        self.metrics.gauge("ws_active", self.metrics.gauges.get("ws_active", 0) + 1)
        await self.outbound.start()
        self.audio = AudioIngest(self.settings, self.metrics, self.recorder, self.outbound.send)

        try:
            with self.recorder.trace(
                "ws.connection",
                client=str(self.ws.client)[:80] if self.ws.client else None,
            ) as span:
                hello = await self._await_hello()
                if hello is None:
                    return
                session, resumed = await self.runtime.sessions.resume_or_create(
                    hello.session_id, hello.client
                )
                self.session = session
                session.kb_version = self.runtime.retriever.kb_version
                span.set(session_id=session.id, resumed=resumed)
                self.metrics.incr("ws_sessions_resumed" if resumed else "ws_sessions_new")

                await self.outbound.send(
                    protocol.ready(
                        session.id,
                        resumed=resumed,
                        kb_version=session.kb_version,
                        model=self.runtime.provider.model,
                    )
                )
                if resumed and session.history:
                    await self.outbound.send(protocol.history_replay(session.history[-20:]))

                self._heartbeat = asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat")
                await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except ConnectionClosed:
            pass
        except Exception:  # noqa: BLE001 - connection errors must not kill the worker
            logger.exception("connection failed")
        finally:
            await self._cleanup()

    # ── handshake ──────────────────────────────────────────────────────────
    async def _await_hello(self) -> HelloFrame | None:
        try:
            raw = await asyncio.wait_for(
                self.ws.receive_json(), timeout=self.settings.ws_hello_timeout_s
            )
        except TimeoutError:
            await self._safe_error("hello timeout", code="hello_timeout")
            await self.ws.close(code=CLOSE_PROTOCOL_ERROR)
            return None
        except (json.JSONDecodeError, ValueError):
            await self._safe_error("hello must be valid JSON", code="bad_frame")
            await self.ws.close(code=CLOSE_PROTOCOL_ERROR)
            return None
        try:
            frame = parse_client_frame(raw if isinstance(raw, dict) else {})
        except FrameError as exc:
            await self._safe_error(str(exc), code=exc.code)
            await self.ws.close(code=CLOSE_PROTOCOL_ERROR)
            return None
        if not isinstance(frame, HelloFrame):
            await self._safe_error("first frame must be `hello`", code="hello_required")
            await self.ws.close(code=CLOSE_PROTOCOL_ERROR)
            return None
        if frame.protocol_version != protocol.PROTOCOL_VERSION:
            await self._safe_error(
                f"unsupported protocol_version {frame.protocol_version}",
                code="protocol_mismatch",
            )
            await self.ws.close(code=CLOSE_PROTOCOL_ERROR)
            return None
        return frame

    # ── receive loop ───────────────────────────────────────────────────────
    async def _receive_loop(self) -> None:
        timeout = self.settings.ws_heartbeat_interval_s * 2.5
        while not self._closing:
            try:
                raw = await asyncio.wait_for(self.ws.receive_json(), timeout=timeout)
            except TimeoutError:
                self.metrics.incr("ws_stale_closed")
                with contextlib.suppress(Exception):
                    await self.ws.close(code=CLOSE_STALE)
                return

            try:
                frame = parse_client_frame(raw if isinstance(raw, dict) else {})
            except FrameError as exc:
                await self.outbound.send(protocol.error(str(exc), code=exc.code))
                self.metrics.incr("ws_bad_frames")
                continue

            if isinstance(frame, PromptFrame):
                await self._start_turn(frame)
            elif isinstance(frame, AbortFrame):
                await self._abort(frame)
            elif isinstance(frame, AudioFrame):
                await self._handle_audio(frame)
            elif isinstance(frame, PingFrame):
                await self.outbound.send(protocol.pong(frame.ts))
            elif isinstance(frame, PongFrame):
                continue
            elif isinstance(frame, HelloFrame):
                await self.outbound.send(
                    protocol.ready(
                        self.session.id if self.session else "",
                        resumed=True,
                        kb_version=self.runtime.retriever.kb_version,
                        model=self.runtime.provider.model,
                    )
                )

    # ── turns ──────────────────────────────────────────────────────────────
    async def _start_turn(self, frame: PromptFrame) -> None:
        assert self.session is not None
        if self.session.active_turn is not None:
            await self._abort_active("superseded")
        handle = TurnHandle(turn_id=frame.id)
        self.session.active_turn = handle
        await self.outbound.send(protocol.ack(frame.id))
        task = asyncio.create_task(self._run_turn(frame, handle), name=f"turn-{frame.id}")
        handle.task = task
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _run_turn(self, frame: PromptFrame, handle: TurnHandle) -> None:
        assert self.session is not None
        assert self.audio is not None
        session = self.session
        request = TurnRequest(
            turn_id=frame.id,
            prompt=frame.text,
            top_k=frame.options.top_k,
            mode=frame.options.mode,
            tools=frame.options.tools,
            temperature=frame.options.temperature,
            max_tokens=frame.options.max_tokens,
        )
        answer: list[str] = []
        done: dict[str, Any] = {}
        status = "error"
        try:
            async for out in self.runtime.agent.run(request, session):
                if out.get("type") == "delta":
                    answer.append(out["text"])
                    await self.outbound.push_delta(handle.turn_id, out["text"])
                else:
                    if out.get("type") == "done":
                        done = out
                        status = "error" if out.get("finish_reason") == "error" else "done"
                    # terminal/control frames must not overtake buffered text
                    await self.outbound.flush_turn(handle.turn_id)
                    await self.outbound.send(out)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            status = "aborted"
            with contextlib.suppress(Exception):
                await self.outbound.flush_turn(handle.turn_id)
                await self.outbound.send(protocol.aborted(handle.turn_id, "cancelled"))
            self.metrics.incr("turns_aborted")
        except Exception as exc:  # noqa: BLE001 - terminal frame for the client
            logger.exception("turn failed")
            self.metrics.incr("turns_failed")
            status = "error"
            with contextlib.suppress(Exception):
                await self.outbound.send(
                    protocol.error(
                        f"{type(exc).__name__}: {exc}",
                        code="internal_error",
                        turn_id=handle.turn_id,
                        retryable=False,
                    )
                )
        finally:
            if session.active_turn is handle:
                session.active_turn = None
            session.turns += 1
            session.kb_version = self.runtime.retriever.kb_version
            # The journal write is a queue push, not an await on Postgres: the
            # turn is already finished, and durability must not delay the next one.
            self.runtime.sessions.record_turn(
                session,
                turn_id=handle.turn_id,
                prompt=frame.text,
                answer="".join(answer),
                status=status,
                finish_reason=done.get("finish_reason"),
                ttft_ms=done.get("ttft_ms"),
                total_ms=done.get("total_ms"),
                tool_iterations=int(done.get("tool_iterations") or 0),
                usage=done.get("usage") or {},
                model=done.get("model"),
            )
            await self.runtime.sessions.save(session)

    async def _abort(self, frame: AbortFrame) -> None:
        if self.session is None:
            return
        active = self.session.active_turn
        if active is None:
            if frame.id:
                await self.outbound.send(protocol.aborted(frame.id, "no_active_turn"))
            return
        if frame.id and frame.id != active.turn_id:
            # may live on another replica: fan out over the bus
            await self.runtime.bus.publish(
                {
                    "kind": "turn.abort",
                    "session_id": self.session.id,
                    "turn_id": frame.id,
                    "origin": self.runtime.instance_id,
                }
            )
            await self.outbound.send(protocol.aborted(frame.id, "not_local"))
            return
        await self._abort_active("client_abort")

    async def _abort_active(self, reason: str) -> None:
        assert self.session is not None
        handle = self.session.active_turn
        if handle is None or handle.task is None:
            return
        handle.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
            await asyncio.wait_for(handle.task, timeout=2.0)
        logger.debug("turn %s aborted (%s)", handle.turn_id, reason)

    # ── audio ──────────────────────────────────────────────────────────────
    async def _handle_audio(self, frame: AudioFrame) -> None:
        assert self.session is not None
        assert self.audio is not None
        if not self.settings.audio_enabled:
            await self.outbound.send(protocol.error("audio ingest disabled", code="audio_disabled"))
            return
        try:
            ack = await self.audio.handle(frame, self.session)
        except ValueError as exc:
            await self.outbound.send(protocol.error(str(exc), code="bad_audio_frame"))
            return
        await self.outbound.send(ack)

    # ── heartbeat ──────────────────────────────────────────────────────────
    async def _heartbeat_loop(self) -> None:
        interval = self.settings.ws_heartbeat_interval_s
        while not self._closing:
            await asyncio.sleep(interval)
            try:
                await self.outbound.send(protocol.ping())
            except ConnectionClosed:
                return

    # ── teardown ───────────────────────────────────────────────────────────
    async def _safe_error(self, message: str, *, code: str) -> None:
        with contextlib.suppress(Exception):
            await self.outbound.send(protocol.error(message, code=code))
            await self.outbound.drain(timeout_s=1.0)

    async def _cleanup(self) -> None:
        self._closing = True
        if self._heartbeat:
            self._heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat
        for task in list(self._turn_tasks):
            task.cancel()
        for task in list(self._turn_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=1.0)
        if self.audio is not None:
            await self.audio.close()
        if self.session is not None:
            self.session.active_turn = None
            with contextlib.suppress(Exception):
                await self.runtime.sessions.save(self.session)
        await self.outbound.close()
        self.metrics.gauge("ws_active", max(0, self.metrics.gauges.get("ws_active", 1) - 1))
        with contextlib.suppress(Exception):
            await self.ws.close()


def authorized(websocket: WebSocket, runtime: Runtime) -> bool:
    token = runtime.settings.api_token
    if not token:
        return True
    supplied = websocket.query_params.get("token")
    return supplied == token


async def handle_connection(websocket: WebSocket, runtime: Runtime) -> None:
    if not authorized(websocket, runtime):
        await websocket.accept()
        await websocket.send_json(
            protocol.error("unauthorized", code="unauthorized", retryable=False)
        )
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return
    connection = Connection(websocket, runtime)
    started = time.perf_counter()
    try:
        await connection.run()
    finally:
        runtime.metrics.observe("ws_connection_ms", (time.perf_counter() - started) * 1000)

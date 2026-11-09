from __future__ import annotations

import asyncio
from typing import Any

from app.telemetry.metrics import Metrics
from app.ws.outbound import OutboundStream
from tests.conftest import make_settings


class FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)


def delta_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "delta"]


async def test_deltas_are_coalesced_and_lossless() -> None:
    ws = FakeWS()
    metrics = Metrics()
    out = OutboundStream(ws, make_settings(delta_flush_ms=5, ws_send_queue_size=128), metrics)
    await out.start()
    try:
        for _ in range(300):
            await out.push_delta("t1", "x")
        await out.flush_turn("t1")
        await asyncio.sleep(0.05)
    finally:
        await out.close()

    deltas = delta_frames(ws.frames)
    assert 1 <= len(deltas) <= 10, f"expected coalescing, got {len(deltas)} frames"
    assert "".join(frame["text"] for frame in deltas) == "x" * 300
    assert sum(frame["coalesced"] for frame in deltas) == 300
    assert [frame["seq"] for frame in deltas] == list(range(1, len(deltas) + 1))


async def test_control_frames_are_ordered_before_done() -> None:
    ws = FakeWS()
    out = OutboundStream(ws, make_settings(delta_flush_ms=5), Metrics())
    await out.start()
    try:
        await out.push_delta("t1", "hello ")
        await out.push_delta("t1", "world")
        await out.flush_turn("t1")
        await out.send({"type": "done", "turn_id": "t1"})
        await asyncio.sleep(0.03)
    finally:
        await out.close()

    types = [frame["type"] for frame in ws.frames]
    assert types.index("delta") < types.index("done")
    text = "".join(f["text"] for f in delta_frames(ws.frames))
    assert text == "hello world"


async def test_queue_full_drops_frames_not_text() -> None:
    ws = FakeWS()
    metrics = Metrics()
    out = OutboundStream(ws, make_settings(ws_send_queue_size=1), metrics)
    # writer intentionally not started: the queue stays full
    await out.send({"type": "ping"})
    for _ in range(5):
        await out.push_delta("t1", "abc")
    await out.flush_turn("t1")
    assert metrics.counters.get("delta_frames_dropped", 0) >= 1

    # drain the control frame, then flush again: no text was lost
    out._queue.get_nowait()
    await out.flush_turn("t1")
    frame = out._queue.get_nowait()
    assert frame["type"] == "delta"
    assert frame["text"] == "abc" * 5

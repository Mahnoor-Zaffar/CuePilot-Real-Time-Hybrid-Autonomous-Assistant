from __future__ import annotations

import asyncio
import base64

from app.main import create_app
from app.telemetry.metrics import Metrics
from app.ws.limits import ClientLimiter
from fastapi.testclient import TestClient
from tests.conftest import drain_ws, make_settings


def handshake(ws) -> dict:
    ws.send_json({"type": "hello", "client": {"name": "pytest", "version": "0"}})
    ready = ws.receive_json()
    assert ready["type"] == "ready"
    return ready


def test_full_turn_streams_retrieval_tool_call_and_tokens(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ready = handshake(ws)
        assert ready["protocol_version"] == 1
        assert ready["model"].startswith("mock/")

        ws.send_json({"type": "prompt", "id": "t1", "text": "What does CUE-E1042 mean?"})
        frames = drain_ws(ws, until="done")
        types = [frame["type"] for frame in frames]

        assert types[0] == "ack"
        assert types.count("ack") == 1
        assert "retrieval" in types
        assert types.count("tool_call_start") == types.count("tool_call_result") == 1
        assert types[-1] == "done"

        retrieval = next(frame for frame in frames if frame["type"] == "retrieval")
        assert retrieval["hits"]
        assert retrieval["hits"][0]["explain"]["contributions"]

        text = "".join(frame["text"] for frame in frames if frame["type"] == "delta")
        assert text
        done = frames[-1]
        assert done["ttft_ms"] is not None
        assert done["usage"]["completion_tokens"] > 0


def test_abort_cancels_in_flight_turn() -> None:
    app = create_app(make_settings(mock_token_delay_ms=40))
    with TestClient(app) as client, client.websocket_connect("/ws/stream") as ws:
        handshake(ws)
        ws.send_json(
            {"type": "prompt", "id": "t-slow", "text": "Explain hybrid retrieval in detail"}
        )
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        ws.send_json({"type": "abort", "id": "t-slow"})
        frames = drain_ws(ws, until="aborted")
        assert frames[-1] == {"type": "aborted", "turn_id": "t-slow", "reason": "cancelled"}

        # the socket stays usable after an abort
        ws.send_json({"type": "ping", "ts": 7})
        pong = ws.receive_json()
        while pong["type"] != "pong":
            pong = ws.receive_json()
        assert pong["ts"] == 7


def test_reconnect_resumes_session_and_replays_history(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ready = handshake(ws)
        session_id = ready["session_id"]
        ws.send_json({"type": "prompt", "id": "t1", "text": "What is CUE-E2041?"})
        drain_ws(ws, until="done")

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "hello", "session_id": session_id})
        resumed = ws.receive_json()
        assert resumed["type"] == "ready"
        assert resumed["resumed"] is True
        assert resumed["session_id"] == session_id

        history = ws.receive_json()
        assert history["type"] == "history"
        roles = [message["role"] for message in history["messages"]]
        assert roles[0] == "user"
        assert "assistant" in roles


def test_unknown_session_id_creates_a_new_one(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "hello", "session_id": "does-not-exist"})
        ready = ws.receive_json()
        assert ready["resumed"] is False
        assert ready["session_id"] != "does-not-exist"


def test_bad_frame_returns_error_and_keeps_socket_open(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        handshake(ws)
        ws.send_json({"type": "definitely-not-a-frame"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "bad_frame"

        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_audio_frames_get_acknowledged(client: TestClient) -> None:
    pcm = b"\x00\x10" * 1600  # 100ms of 16kHz mono s16le
    with client.websocket_connect("/ws/stream") as ws:
        handshake(ws)
        ws.send_json(
            {"type": "audio", "seq": 1, "pcm": base64.b64encode(pcm).decode(), "sample_rate": 16000}
        )
        ack = ws.receive_json()
        assert ack["type"] == "audio_ack"
        assert ack["seq"] == 1
        assert 80 <= ack["buffered_ms"] <= 120
        assert ack["rms"] > 0


def test_first_frame_must_be_hello(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "prompt", "id": "t1", "text": "hi"})
        error = ws.receive_json()
        assert error["code"] == "hello_required"


def test_rate_limit_rejects_connections_over_the_window() -> None:
    app = create_app(make_settings(rate_limit_connect_attempts=1))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert handshake(ws)["type"] == "ready"

        with client.websocket_connect("/ws/stream") as ws:
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert frame["code"] == "client_rate_limited"
            assert frame["retryable"] is True
            assert frame["retry_after_ms"] >= 100


def test_connection_cap_rejects_a_second_socket_on_one_replica() -> None:
    app = create_app(
        make_settings(rate_limit_max_connections=1, rate_limit_connect_attempts=100)
    )
    with TestClient(app) as client, client.websocket_connect("/ws/stream") as first:
        handshake(first)
        with client.websocket_connect("/ws/stream") as second:
            frame = second.receive_json()
            assert frame["type"] == "error"
            assert frame["code"] == "connection_cap"


async def test_a_concurrent_admit_cannot_overrun_the_cap() -> None:
    class SlowCache:
        backend = "slow"

        async def incr(self, key: str, ttl_s: int | None = None) -> int:
            await asyncio.sleep(0.01)
            return 1

    metrics = Metrics()
    limiter = ClientLimiter(
        SlowCache(),  # type: ignore[arg-type]
        make_settings(rate_limit_max_connections=1),
        metrics,
    )

    first, second = await asyncio.gather(limiter.admit("a"), limiter.admit("a"))

    # The first call reserved the only slot before its await, so the second is
    # rejected even though the gauge has not moved yet.
    assert first.allowed is True
    assert second.allowed is False
    assert second.code == "connection_cap"
    assert limiter.reserved == 1
    limiter.release_reservation()
    assert limiter.reserved == 0


async def test_limiter_fails_open_when_the_cache_is_unavailable() -> None:
    class BrokenCache:
        backend = "broken"

        async def incr(self, key: str, ttl_s: int | None = None) -> int:
            raise ConnectionError("redis went away")

    metrics = Metrics()
    limiter = ClientLimiter(BrokenCache(), make_settings(), metrics)  # type: ignore[arg-type]

    decision = await limiter.admit("127.0.0.1")

    assert decision.allowed is True
    assert metrics.counters["rate_limit_cache_errors"] == 1


def test_websocket_auth_rejects_bad_token() -> None:
    app = create_app(make_settings(api_token="secret"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream?token=wrong") as ws:
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert frame["code"] == "unauthorized"

        with client.websocket_connect("/ws/stream?token=secret") as ws:
            assert handshake(ws)["type"] == "ready"

"""Wire protocol: typed client frames, server frame builders.

Client frames are a discriminated union on `type`. Anything that fails validation
gets a `bad_frame` error instead of a silent drop, so version skew is loud.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

PROTOCOL_VERSION = 1


# ── client -> server ────────────────────────────────────────────────────────
class PromptOptions(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    tools: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=16, le=8000)


class HelloFrame(BaseModel):
    type: Literal["hello"]
    session_id: str | None = None
    protocol_version: int = PROTOCOL_VERSION
    client: dict[str, Any] = Field(default_factory=dict)


class PromptFrame(BaseModel):
    type: Literal["prompt"]
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=8000)
    options: PromptOptions = Field(default_factory=PromptOptions)


class AbortFrame(BaseModel):
    type: Literal["abort"]
    id: str | None = Field(default=None, max_length=128)


class AudioFrame(BaseModel):
    type: Literal["audio"]
    seq: int = Field(ge=0)
    pcm: str = Field(min_length=4, max_length=1_400_000)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)


class PingFrame(BaseModel):
    type: Literal["ping"]
    ts: int | None = None


class PongFrame(BaseModel):
    type: Literal["pong"]
    ts: int | None = None


ClientFrame = Annotated[
    HelloFrame | PromptFrame | AbortFrame | AudioFrame | PingFrame | PongFrame,
    Field(discriminator="type"),
]

_client_adapter: TypeAdapter[ClientFrame] = TypeAdapter(ClientFrame)


class FrameError(ValueError):
    def __init__(self, message: str, *, code: str = "bad_frame") -> None:
        super().__init__(message)
        self.code = code


def parse_client_frame(raw: dict[str, Any]) -> ClientFrame:
    try:
        return _client_adapter.validate_python(raw)
    except Exception as exc:  # noqa: BLE001 - normalised into a protocol error
        raise FrameError(str(exc)[:300]) from exc


# ── server -> client ────────────────────────────────────────────────────────
def ready(session_id: str, *, resumed: bool, kb_version: int, model: str) -> dict[str, Any]:
    return {
        "type": "ready",
        "session_id": session_id,
        "resumed": resumed,
        "protocol_version": PROTOCOL_VERSION,
        "kb_version": kb_version,
        "model": model,
        "server_time": int(time.time() * 1000),
    }


def ack(turn_id: str) -> dict[str, Any]:
    return {"type": "ack", "turn_id": turn_id, "ts": int(time.time() * 1000)}


def delta(turn_id: str, seq: int, text: str, *, coalesced: int = 1) -> dict[str, Any]:
    return {"type": "delta", "turn_id": turn_id, "seq": seq, "text": text, "coalesced": coalesced}


def aborted(turn_id: str, reason: str) -> dict[str, Any]:
    return {"type": "aborted", "turn_id": turn_id, "reason": reason}


def error(
    message: str,
    *,
    code: str = "internal_error",
    turn_id: str | None = None,
    retryable: bool = False,
    retry_after_ms: int | None = None,
) -> dict[str, Any]:
    payload = {
        "type": "error",
        "code": code,
        "message": message[:600],
        "retryable": retryable,
        "ts": int(time.time() * 1000),
    }
    if turn_id:
        payload["turn_id"] = turn_id
    if retry_after_ms is not None:
        payload["retry_after_ms"] = retry_after_ms
    return payload


def pong(ts: int | None) -> dict[str, Any]:
    return {"type": "pong", "ts": ts, "server_time": int(time.time() * 1000)}


def ping() -> dict[str, Any]:
    return {"type": "ping", "ts": int(time.time() * 1000)}


def audio_ack(seq: int, buffered_ms: int, rms: float) -> dict[str, Any]:
    return {"type": "audio_ack", "seq": seq, "buffered_ms": buffered_ms, "rms": round(rms, 4)}


def transcript(text: str, *, final: bool, window_ms: int) -> dict[str, Any]:
    return {
        "type": "transcript",
        "text": text,
        "final": final,
        "window_ms": window_ms,
        "ts": int(time.time() * 1000),
    }


def history_replay(history: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "history", "messages": history}

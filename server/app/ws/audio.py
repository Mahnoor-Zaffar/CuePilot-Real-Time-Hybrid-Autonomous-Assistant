"""Raw PCM ingest: level metering now, rolling-window Whisper transcription when
an OpenAI key is configured. Audio never blocks the receive loop."""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.config import Settings
from app.telemetry.metrics import Metrics
from app.telemetry.otel import TraceRecorder
from app.ws.protocol import AudioFrame
from app.ws.session import Session

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], Awaitable[None]]


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


class AudioIngest:
    def __init__(
        self,
        settings: Settings,
        metrics: Metrics,
        recorder: TraceRecorder,
        emit: Emit,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self.recorder = recorder
        self.emit = emit
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task] = set()

    @property
    def transcription_enabled(self) -> bool:
        return bool(self.settings.openai_api_key) and self.settings.audio_enabled

    async def handle(self, frame: AudioFrame, session: Session) -> dict[str, Any]:
        import base64

        try:
            pcm = base64.b64decode(frame.pcm, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid base64 pcm: {exc}") from exc

        session.audio.append(frame.seq, pcm, frame.sample_rate)
        self.metrics.incr("audio_frames_total")
        self.metrics.incr("audio_bytes_total", len(pcm))

        if (
            self.transcription_enabled
            and session.audio.buffered_ms >= self.settings.audio_window_ms
            and not self._transcribing(session.id)
        ):
            task = asyncio.create_task(self._transcribe(session), name="whisper-window")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        return {
            "type": "audio_ack",
            "seq": frame.seq,
            "buffered_ms": session.audio.buffered_ms,
            "rms": round(session.audio.rms(), 4),
            "transcribing": self.transcription_enabled,
        }

    def _transcribing(self, session_id: str) -> bool:
        return any(not t.done() and t.get_name() == "whisper-window" for t in self._tasks)

    async def _transcribe(self, session: Session) -> None:
        pcm = session.audio.take()
        if not pcm:
            return
        window_ms = session.audio.buffered_ms
        try:
            with self.recorder.span("audio.transcribe", window_ms=window_ms) as span:
                client = await self._get_client()
                response = await client.post(
                    "/audio/transcriptions",
                    files={"file": ("window.wav", wav_bytes(pcm, session.audio.sample_rate), "audio/wav")},
                    data={"model": self.settings.transcription_model, "response_format": "json"},
                )
                response.raise_for_status()
                text = (response.json().get("text") or "").strip()
                span.set(chars=len(text))
            if text:
                await self.emit({"type": "transcript", "text": text, "final": True, "window_ms": window_ms})
        except Exception as exc:  # noqa: BLE001 - transcription is best effort
            logger.info("transcription failed: %s", exc)
            self.metrics.incr("transcription_failures")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.transcription_base_url.rstrip("/"),
                timeout=self.settings.embedding_timeout_s,
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            )
        return self._client

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

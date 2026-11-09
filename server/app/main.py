from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, kb, observability
from app.config import Settings, get_settings
from app.runtime import Runtime
from app.ws.connection import handle_connection


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        )
    logging.getLogger("app").setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = await Runtime.create(settings)
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="cuePilot Gateway",
        version="0.1.0",
        description=(
            "Bi-directional WebSocket gateway: hybrid BM25 + pgvector retrieval, "
            "RRF fusion, streaming tool-calling agent, OpenTelemetry traces."
        ),
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(kb.router)
    app.include_router(observability.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "cuepilot-gateway",
            "docs": "/docs",
            "ws": "/ws/stream",
            "health": "/api/health",
        }

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        runtime: Runtime | None = getattr(websocket.app.state, "runtime", None)
        if runtime is None:  # pragma: no cover - lifespan guarantees this
            await websocket.close(code=1011)
            return
        await handle_connection(websocket, runtime)

    return app


app = create_app()

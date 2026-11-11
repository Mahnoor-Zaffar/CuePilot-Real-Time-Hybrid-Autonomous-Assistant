from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from app.api.deps import get_runtime, require_auth
from app.runtime import Runtime
from app.telemetry.metrics import render_prometheus

router = APIRouter(prefix="/api", tags=["observability"], dependencies=[Depends(require_auth)])

# Prometheus scrapes `/metrics`, not `/api/metrics`. Same auth dependency, same
# merged snapshot; only the serialisation differs.
prometheus_router = APIRouter(
    tags=["observability"], dependencies=[Depends(require_auth)]
)


@prometheus_router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(runtime: Runtime = Depends(get_runtime)) -> str:
    snapshot = await runtime.aggregated_metrics()
    return render_prometheus(snapshot, runtime.instance_id)


@router.get("/metrics")
async def metrics(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    snapshot = await runtime.aggregated_metrics()
    snapshot["backends"] = {
        "cache": runtime.cache.backend,
        "bus": runtime.bus.backend,
        "vector_store": runtime.store.backend,
        "embedder": runtime.retriever.embedder.name,
        "provider": runtime.provider.name,
    }
    snapshot["kb_version"] = runtime.retriever.kb_version
    return snapshot


@router.get("/traces")
async def traces(
    limit: int = Query(default=20, ge=1, le=200),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return {"traces": runtime.recorder.recent(limit)}


@router.get("/traces/{trace_id}")
async def trace(trace_id: str, runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    tree = runtime.recorder.get(trace_id)
    if tree is None:
        return {"trace": None, "error": "trace not found or evicted"}
    return {"trace": tree}

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_runtime, require_auth
from app.runtime import Runtime

router = APIRouter(prefix="/api", tags=["observability"], dependencies=[Depends(require_auth)])


@router.get("/metrics")
async def metrics(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    snapshot = runtime.metrics.snapshot()
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

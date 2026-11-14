from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from app.api.deps import get_runtime, require_auth
from app.runtime import Runtime
from app.telemetry.metrics import BUCKETS_MS, render_prometheus

router = APIRouter(prefix="/api", tags=["observability"], dependencies=[Depends(require_auth)])

# Prometheus scrapes `/metrics`, not `/api/metrics`. It renders **this replica**
# only: Prometheus is the aggregation layer, so the gateway does not pretend to
# compute exact fleet-wide quantiles in-process.
prometheus_router = APIRouter(
    tags=["observability"], dependencies=[Depends(require_auth)]
)


def _local_snapshot(runtime: Runtime) -> dict[str, Any]:
    snapshot = runtime.metrics.snapshot()
    snapshot["instance_id"] = runtime.instance_id
    snapshot["aggregation"] = {
        "mode": "per_instance",
        "instance_id": runtime.instance_id,
        "quantile_method": "cumulative_buckets",
        "bucket_bounds_ms": list(BUCKETS_MS),
        "note": (
            "this response describes one replica; scrape /metrics from every replica "
            "and aggregate in Prometheus. Histograms are cumulative buckets, so "
            "percentiles are bucket-resolution, not exact"
        ),
    }
    return snapshot


@prometheus_router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(runtime: Runtime = Depends(get_runtime)) -> str:
    return render_prometheus(runtime.metrics.snapshot(), runtime.instance_id)


@router.get("/metrics")
async def metrics(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    snapshot = _local_snapshot(runtime)
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

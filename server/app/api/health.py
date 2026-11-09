from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    """Unauthenticated by design: load balancers must be able to probe it."""
    runtime = request.app.state.runtime
    return await runtime.health()

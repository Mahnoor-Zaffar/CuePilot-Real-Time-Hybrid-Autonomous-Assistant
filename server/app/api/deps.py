from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - only before lifespan
        raise HTTPException(status_code=503, detail="runtime not ready")
    return runtime


async def require_auth(request: Request) -> None:
    """Optional bearer auth: enforced only when api_token is configured."""
    runtime = get_runtime(request)
    token = runtime.settings.api_token
    if not token:
        return
    supplied = request.headers.get("authorization", "")
    if supplied.removeprefix("Bearer ").strip() != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

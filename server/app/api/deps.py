from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.auth import AuthError
from app.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - only before lifespan
        raise HTTPException(status_code=503, detail="runtime not ready")
    return runtime


async def require_auth(request: Request) -> None:
    """Optional auth: JWT when configured, otherwise a static bearer token.

    A configured JWT secret or JWKS URL makes JWT verification mandatory; there
    is no fallback to the static token, because a deployment that turned on
    identity should not silently accept a shared secret instead.
    """
    runtime = get_runtime(request)
    settings = runtime.settings
    if runtime.jwt.enabled:
        try:
            await runtime.jwt.authenticate(
                request.headers.get("authorization"),
                surface="http",
                remote=request.client.host if request.client else None,
                path=request.url.path,
            )
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        return

    token = settings.api_token
    if not token:
        return
    supplied = request.headers.get("authorization", "")
    if supplied.removeprefix("Bearer ").strip() != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

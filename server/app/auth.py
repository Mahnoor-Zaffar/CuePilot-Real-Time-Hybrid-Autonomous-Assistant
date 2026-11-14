"""Optional JWT authentication, verified with the standard library only.

The gateway is often deployed behind a proxy that already authenticated the
user, but "trust a header" is not identity: anyone who can reach the port can
set it. When `auth_jwt_secret` is configured, the gateway verifies an HS256
(HS384/HS512) bearer token itself and uses the configured claim as the
rate-limit identity. No JWT dependency is added for that.

Only HMAC algorithms are accepted on purpose. `alg: none` and asymmetric
algorithms are rejected rather than silently trusted: supporting RS256 without
a key-management story would be a false sense of security.

Verification checks, in order:

* the token has three base64url segments and a JSON object header/payload;
* `alg` is in `auth_jwt_algorithms` **and** is an HS family algorithm;
* the HMAC signature matches (constant-time compare);
* `exp` / `nbf` are respected with `auth_jwt_leeway_s`;
* `aud` / `iss` match when configured;
* the identity claim is a non-empty string.

Anything else raises `AuthError`. The caller turns that into a 401 (HTTP) or a
4401 close (WebSocket); there is no "fail open" path for authentication.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings

_DIGESTS = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}


class AuthError(ValueError):
    """A token that must not be accepted. The message is safe to log."""

    def __init__(self, message: str, *, code: str = "unauthorized") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Identity:
    subject: str
    claims: dict[str, Any]


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _decode_json(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(_b64decode(value))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("token is not valid base64url JSON") from exc
    if not isinstance(decoded, dict):
        raise AuthError("token header and payload must be JSON objects")
    return decoded


def decode_jwt(token: str, settings: Settings, *, now: float | None = None) -> Identity:
    """Verify an HS* JWT and return its identity claim."""
    secret = settings.auth_jwt_secret
    if not secret:
        raise AuthError("jwt auth is not configured")

    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("token is not a JWT")
    header_raw, payload_raw, signature_raw = parts

    header = _decode_json(header_raw)
    claims = _decode_json(payload_raw)
    try:
        signature = _b64decode(signature_raw)
    except ValueError as exc:
        raise AuthError("token signature is not valid base64url") from exc

    alg = str(header.get("alg") or "")
    if alg not in settings.auth_jwt_algorithms:
        raise AuthError(f"token algorithm {alg!r} is not allowed")
    digest = _DIGESTS.get(alg)
    if digest is None:
        # `none` and asymmetric algorithms are explicitly not supported. Reject
        # rather than verify the wrong thing.
        raise AuthError(f"token algorithm {alg!r} is not supported")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_raw}.{payload_raw}".encode("ascii"),
        digest,
    ).digest()
    if not hmac.compare_digest(expected, signature):
        raise AuthError("token signature does not match")

    current = time.time() if now is None else now
    leeway = max(0, settings.auth_jwt_leeway_s)
    exp = claims.get("exp")
    if exp is not None:
        try:
            expired = float(exp) + leeway < current
        except (TypeError, ValueError) as exc:
            raise AuthError("token exp is not a number") from exc
        if expired:
            raise AuthError("token has expired")
    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            too_early = float(nbf) - leeway > current
        except (TypeError, ValueError) as exc:
            raise AuthError("token nbf is not a number") from exc
        if too_early:
            raise AuthError("token is not valid yet")

    if settings.auth_jwt_audience:
        raw_aud = claims.get("aud")
        audiences = raw_aud if isinstance(raw_aud, list) else [raw_aud]
        if settings.auth_jwt_audience not in {str(a) for a in audiences if a is not None}:
            raise AuthError("token audience does not match")
    if settings.auth_jwt_issuer and str(claims.get("iss") or "") != settings.auth_jwt_issuer:
        raise AuthError("token issuer does not match")

    subject = claims.get(settings.auth_jwt_identity_claim)
    if not isinstance(subject, str) or not subject.strip():
        raise AuthError(
            f"token has no usable {settings.auth_jwt_identity_claim!r} claim"
        )
    return Identity(subject=subject.strip()[:128], claims=claims)


def bearer_token(value: str | None) -> str | None:
    """Extract a bearer token, or None when the header is absent/malformed."""
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def authenticate(authorization: str | None, settings: Settings) -> Identity | None:
    """Verify a bearer JWT when configured. Returns None when auth is disabled."""
    if not settings.auth_jwt_secret:
        return None
    token = bearer_token(authorization)
    if token is None:
        raise AuthError("missing bearer token")
    return decode_jwt(token, settings)

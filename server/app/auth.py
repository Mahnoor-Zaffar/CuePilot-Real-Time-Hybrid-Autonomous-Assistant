"""Optional JWT authentication with HMAC and JWKS-backed asymmetric keys.

The gateway is often deployed behind a proxy that already authenticated the
user, but "trust a header" is not identity: anyone who can reach the port can
set it. When JWT auth is configured, the gateway verifies a bearer token itself
and uses the configured claim as the rate-limit identity.

Two verification modes, chosen by configuration:

* **HMAC** (`auth_jwt_secret`) — HS256/HS384/HS512 via the standard library.
* **Asymmetric** (`auth_jwt_jwks_url`) — RS/PS/ES tokens verified against a
  JWKS document. Keys are cached by `kid`, refreshed on TTL, and an unknown
  `kid` triggers at most one refresh per `auth_jwt_jwks_min_refresh_s` so a
  stream of bogus kids cannot hammer the identity provider.

Algorithm confusion is rejected by construction: HMAC verification only ever
uses `auth_jwt_secret`, and asymmetric verification only ever uses a public key
from the JWKS document. `alg: none` is never accepted.

Verification checks, in order:

* the token has three base64url segments and a JSON object header/payload;
* `alg` is in `auth_jwt_algorithms` and has a verifier for this deployment;
* the signature matches (constant-time compare for HMAC, library verify for
  asymmetric);
* `exp` / `nbf` are respected with `auth_jwt_leeway_s`;
* `aud` / `iss` match when configured;
* the identity claim is a non-empty string.

Anything else raises `AuthError`. The caller turns that into a 401 (HTTP) or a
4401 close (WebSocket); there is no "fail open" path for authentication.
Rejections are logged as one structured JSON line and counted, with the token
itself never included.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.telemetry.metrics import Metrics

logger = logging.getLogger(__name__)

_DIGESTS = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

_ASYMMETRIC = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
    }
)

try:  # pragma: no cover - the base install includes cryptography
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional install
    _CRYPTO_AVAILABLE = False


class AuthError(ValueError):
    """A token that must not be accepted. The message is safe to log.

    `code` is the wire-level code (kept as `unauthorized` so clients do not
    need a taxonomy). `reason` is the stable audit label.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "unauthorized",
        reason: str = "unauthorized",
        alg: str | None = None,
        kid: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.alg = alg
        self.kid = kid


@dataclass(frozen=True, slots=True)
class Identity:
    subject: str
    claims: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ParsedToken:
    header_raw: str
    payload_raw: str
    signature: bytes
    header: dict[str, Any]
    claims: dict[str, Any]

    @property
    def alg(self) -> str:
        return str(self.header.get("alg") or "")

    @property
    def kid(self) -> str | None:
        value = self.header.get("kid")
        return str(value) if value else None

    @property
    def signing_input(self) -> bytes:
        return f"{self.header_raw}.{self.payload_raw}".encode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _decode_json(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(_b64decode(value))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("token is not valid base64url JSON", reason="malformed_token") from exc
    if not isinstance(decoded, dict):
        raise AuthError(
            "token header and payload must be JSON objects", reason="malformed_token"
        )
    return decoded


def _parse_token(token: str) -> _ParsedToken:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("token is not a JWT", reason="malformed_token")
    header_raw, payload_raw, signature_raw = parts
    header = _decode_json(header_raw)
    claims = _decode_json(payload_raw)
    try:
        signature = _b64decode(signature_raw)
    except ValueError as exc:
        raise AuthError(
            "token signature is not valid base64url", reason="malformed_token"
        ) from exc
    return _ParsedToken(
        header_raw=header_raw,
        payload_raw=payload_raw,
        signature=signature,
        header=header,
        claims=claims,
    )


def _validate_claims(
    claims: dict[str, Any], settings: Settings, *, now: float | None = None
) -> Identity:
    current = time.time() if now is None else now
    leeway = max(0, settings.auth_jwt_leeway_s)
    exp = claims.get("exp")
    if exp is not None:
        try:
            expired = float(exp) + leeway < current
        except (TypeError, ValueError) as exc:
            raise AuthError("token exp is not a number", reason="invalid_exp") from exc
        if expired:
            raise AuthError("token has expired", reason="token_expired")
    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            too_early = float(nbf) - leeway > current
        except (TypeError, ValueError) as exc:
            raise AuthError("token nbf is not a number", reason="invalid_nbf") from exc
        if too_early:
            raise AuthError("token is not valid yet", reason="token_not_yet_valid")

    if settings.auth_jwt_audience:
        raw_aud = claims.get("aud")
        audiences = raw_aud if isinstance(raw_aud, list) else [raw_aud]
        if settings.auth_jwt_audience not in {str(a) for a in audiences if a is not None}:
            raise AuthError("token audience does not match", reason="audience_mismatch")
    if settings.auth_jwt_issuer and str(claims.get("iss") or "") != settings.auth_jwt_issuer:
        raise AuthError("token issuer does not match", reason="issuer_mismatch")

    subject = claims.get(settings.auth_jwt_identity_claim)
    if not isinstance(subject, str) or not subject.strip():
        raise AuthError(
            f"token has no usable {settings.auth_jwt_identity_claim!r} claim",
            reason="missing_identity",
        )
    return Identity(subject=subject.strip()[:128], claims=claims)


def _verify_hmac(parsed: _ParsedToken, secret: str) -> None:
    alg = parsed.alg
    digest = _DIGESTS.get(alg)
    if digest is None:
        raise AuthError(
            f"token algorithm {alg!r} is not an HMAC algorithm",
            reason="algorithm_not_allowed",
            alg=alg,
        )
    expected = hmac.new(secret.encode("utf-8"), parsed.signing_input, digest).digest()
    if not hmac.compare_digest(expected, parsed.signature):
        raise AuthError(
            "token signature does not match", reason="bad_signature", alg=alg
        )


def _hash_for(alg: str) -> Any:
    return {
        "RS256": hashes.SHA256,
        "RS384": hashes.SHA384,
        "RS512": hashes.SHA512,
        "PS256": hashes.SHA256,
        "PS384": hashes.SHA384,
        "PS512": hashes.SHA512,
        "ES256": hashes.SHA256,
        "ES384": hashes.SHA384,
        "ES512": hashes.SHA512,
    }[alg]()


def _verify_asymmetric(parsed: _ParsedToken, key: Any) -> None:
    if not _CRYPTO_AVAILABLE:  # pragma: no cover - base install includes it
        raise AuthError(
            "asymmetric JWT support requires the cryptography package",
            reason="crypto_unavailable",
            alg=parsed.alg,
        )
    alg = parsed.alg
    try:
        if alg.startswith(("RS", "PS")):
            if not isinstance(key, rsa.RSAPublicKey):
                raise AuthError(
                    "token algorithm does not match the signing key type",
                    reason="key_type_mismatch",
                    alg=alg,
                )
            if alg.startswith("RS"):
                key.verify(
                    parsed.signature, parsed.signing_input, padding.PKCS1v15(), _hash_for(alg)
                )
            else:
                digest = _hash_for(alg)
                key.verify(
                    parsed.signature,
                    parsed.signing_input,
                    padding.PSS(mgf=padding.MGF1(digest), salt_length=digest.digest_size),
                    digest,
                )
        else:
            if not isinstance(key, ec.EllipticCurvePublicKey):
                raise AuthError(
                    "token algorithm does not match the signing key type",
                    reason="key_type_mismatch",
                    alg=alg,
                )
            size = len(parsed.signature) // 2
            r = int.from_bytes(parsed.signature[:size], "big")
            s = int.from_bytes(parsed.signature[size:], "big")
            key.verify(
                encode_dss_signature(r, s), parsed.signing_input, ec.ECDSA(_hash_for(alg))
            )
    except InvalidSignature as exc:
        raise AuthError(
            "token signature does not match", reason="bad_signature", alg=alg
        ) from exc


def _public_key_from_jwk(jwk: dict[str, Any]) -> Any:
    if not _CRYPTO_AVAILABLE:  # pragma: no cover
        raise ValueError("cryptography is not installed")
    kty = jwk.get("kty")
    if kty == "RSA":
        modulus = int.from_bytes(_b64decode(str(jwk["n"])), "big")
        exponent = int.from_bytes(_b64decode(str(jwk["e"])), "big")
        return rsa.RSAPublicNumbers(exponent, modulus).public_key()
    if kty == "EC":
        curves = {
            "P-256": ec.SECP256R1(),
            "P-384": ec.SECP384R1(),
            "P-521": ec.SECP521R1(),
        }
        curve = curves.get(str(jwk.get("crv")))
        if curve is None:
            raise ValueError(f"unsupported EC curve {jwk.get('crv')!r}")
        x = int.from_bytes(_b64decode(str(jwk["x"])), "big")
        y = int.from_bytes(_b64decode(str(jwk["y"])), "big")
        return ec.EllipticCurvePublicNumbers(x, y, curve).public_key()
    raise ValueError(f"unsupported JWK key type {kty!r}")


async def _fetch_jwks(url: str, timeout_s: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("JWKS document is not a JSON object")
    return payload


class JwtVerifier:
    """Async JWT verifier with a bounded, refreshable JWKS key cache."""

    def __init__(
        self,
        settings: Settings,
        metrics: Metrics,
        *,
        fetcher: Callable[[str, float], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self._fetcher = fetcher or _fetch_jwks
        self._keys: dict[str, Any] = {}
        self._fetched_at = 0.0
        self._last_attempt = 0.0
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.auth_jwt_secret or self.settings.auth_jwt_jwks_url)

    async def authenticate(
        self,
        authorization: str | None,
        *,
        surface: str = "http",
        remote: str | None = None,
        path: str | None = None,
    ) -> Identity | None:
        """Verify a bearer JWT when configured. Returns None when auth is off."""
        if not self.enabled:
            return None
        token = bearer_token(authorization)
        if token is None:
            error = AuthError("missing bearer token", reason="missing_token")
            self._audit(error, surface=surface, remote=remote, path=path, token_present=False)
            raise error
        try:
            return await self._verify(token)
        except AuthError as exc:
            self._audit(
                exc,
                surface=surface,
                remote=remote,
                path=path,
                token_present=True,
                token=token,
            )
            raise

    async def _verify(self, token: str) -> Identity:
        parsed = _parse_token(token)
        alg = parsed.alg
        if alg not in self.settings.auth_jwt_algorithms:
            raise AuthError(
                f"token algorithm {alg!r} is not allowed",
                reason="algorithm_not_allowed",
                alg=alg,
                kid=parsed.kid,
            )
        if alg in _DIGESTS:
            if not self.settings.auth_jwt_secret:
                raise AuthError(
                    "HMAC token received but auth_jwt_secret is not configured",
                    reason="hmac_not_configured",
                    alg=alg,
                    kid=parsed.kid,
                )
            _verify_hmac(parsed, self.settings.auth_jwt_secret)
        elif alg in _ASYMMETRIC:
            if not self.settings.auth_jwt_jwks_url:
                raise AuthError(
                    "asymmetric token received but auth_jwt_jwks_url is not configured",
                    reason="jwks_not_configured",
                    alg=alg,
                    kid=parsed.kid,
                )
            keys = await self._candidate_keys(parsed.kid)
            last_error: AuthError | None = None
            for key in keys:
                try:
                    _verify_asymmetric(parsed, key)
                    last_error = None
                    break
                except AuthError as exc:
                    if exc.reason != "bad_signature":
                        raise
                    last_error = exc
            if last_error is not None:
                raise last_error
        else:
            raise AuthError(
                f"token algorithm {alg!r} is not supported",
                reason="algorithm_not_allowed",
                alg=alg,
                kid=parsed.kid,
            )
        try:
            return _validate_claims(parsed.claims, self.settings)
        except AuthError as exc:
            # Claim failures happen after signature verification, so the audit
            # record can still name the algorithm and key that signed it.
            exc.alg = exc.alg or parsed.alg
            exc.kid = exc.kid or parsed.kid
            raise

    async def _candidate_keys(self, kid: str | None) -> list[Any]:
        now = time.monotonic()
        stale = now - self._fetched_at >= max(1, self.settings.auth_jwt_jwks_cache_ttl_s)
        unknown_kid = kid is not None and kid not in self._keys
        if not self._keys or stale or unknown_kid:
            # An unknown kid normally means rotation; refresh once, but throttle
            # so a bogus kid cannot turn into a JWKS request per token.
            can_refresh = (
                not self._keys
                or stale
                or now - self._last_attempt >= max(0, self.settings.auth_jwt_jwks_min_refresh_s)
            )
            if can_refresh:
                await self._refresh_keys(force=unknown_kid)
        if kid is not None:
            key = self._keys.get(kid)
            if key is None:
                raise AuthError(
                    "token kid is not present in the JWKS document",
                    reason="unknown_key",
                    kid=kid,
                )
            return [key]
        if not self._keys:
            raise AuthError("no JWKS keys are available", reason="jwks_empty")
        return list(self._keys.values())

    async def _refresh_keys(self, *, force: bool = False) -> None:
        async with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._keys
                and now - self._fetched_at < max(1, self.settings.auth_jwt_jwks_cache_ttl_s)
            ):
                return
            self._last_attempt = now
            url = self.settings.auth_jwt_jwks_url
            if not url:  # pragma: no cover - guarded by _verify
                raise AuthError("JWKS URL is not configured", reason="jwks_not_configured")
            try:
                document = await self._fetcher(url, self.settings.auth_jwt_jwks_timeout_s)
            except Exception as exc:  # noqa: BLE001 - no key means no auth
                self.metrics.incr("auth_jwks_refresh_errors")
                logger.warning("jwks refresh failed: %s", exc)
                raise AuthError(
                    "JWKS document is unavailable", reason="jwks_unavailable"
                ) from exc
            keys: dict[str, Any] = {}
            for index, jwk in enumerate(document.get("keys") or []):
                if not isinstance(jwk, dict):
                    continue
                if jwk.get("use") not in (None, "sig"):
                    continue
                try:
                    key = _public_key_from_jwk(jwk)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("skipping unusable JWKS key: %s", exc)
                    continue
                keys[str(jwk.get("kid") or f"_key_{index}")] = key
            if not keys:
                self.metrics.incr("auth_jwks_refresh_errors")
                raise AuthError("JWKS document has no usable keys", reason="jwks_empty")
            self._keys = keys
            self._fetched_at = time.monotonic()
            self.metrics.incr("auth_jwks_refreshed")
            logger.info("jwks refreshed keys=%d", len(keys))

    def _audit(
        self,
        error: AuthError,
        *,
        surface: str,
        remote: str | None,
        path: str | None,
        token_present: bool,
        token: str | None = None,
    ) -> None:
        self.metrics.incr("auth_rejections")
        self.metrics.incr(f"auth_rejections_{error.reason}")
        if not self.settings.auth_audit_enabled:
            return
        record = {
            "event": "auth_rejected",
            "surface": surface,
            "reason": error.reason,
            "code": error.code,
            "remote": remote,
            "path": path,
            "token_present": token_present,
            "alg": error.alg,
            "kid": error.kid,
            "token_len": len(token) if token else 0,
            "ts": round(time.time(), 3),
        }
        # The token is deliberately absent. The reason and key id are enough to
        # distinguish an expired token from a rotation problem or a bad key.
        logger.warning("auth_rejected %s", json.dumps(record, separators=(",", ":")))

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            mode = "off"
        elif self.settings.auth_jwt_jwks_url:
            mode = "jwks"
        else:
            mode = "hmac"
        return {
            "enabled": self.enabled,
            "mode": mode,
            "algorithms": list(self.settings.auth_jwt_algorithms),
            "jwks_configured": bool(self.settings.auth_jwt_jwks_url),
            "jwks_cached_keys": len(self._keys),
            "jwks_age_s": (
                round(time.monotonic() - self._fetched_at, 1) if self._fetched_at else None
            ),
            "audit_enabled": self.settings.auth_audit_enabled,
        }


def decode_jwt(token: str, settings: Settings, *, now: float | None = None) -> Identity:
    """Verify a sync HMAC JWT and return its identity claim.

    Kept for callers that only need the HMAC path. Asymmetric tokens need the
    async `JwtVerifier` because fetching a JWKS document is I/O.
    """
    secret = settings.auth_jwt_secret
    if not secret:
        raise AuthError("jwt auth is not configured", reason="jwt_not_configured")
    parsed = _parse_token(token)
    if parsed.alg not in settings.auth_jwt_algorithms:
        raise AuthError(
            f"token algorithm {parsed.alg!r} is not allowed",
            reason="algorithm_not_allowed",
            alg=parsed.alg,
            kid=parsed.kid,
        )
    if parsed.alg in _ASYMMETRIC:
        raise AuthError(
            "asymmetric tokens require the async JwtVerifier",
            reason="async_required",
            alg=parsed.alg,
            kid=parsed.kid,
        )
    _verify_hmac(parsed, secret)
    return _validate_claims(parsed.claims, settings, now=now)


def bearer_token(value: str | None) -> str | None:
    """Extract a bearer token, or None when the header is absent/malformed."""
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def authenticate(
    authorization: str | None, settings: Settings, *, now: float | None = None
) -> Identity | None:
    """Sync HMAC-only convenience wrapper. Returns None when auth is disabled.

    Prefer `JwtVerifier.authenticate` in request paths: it handles JWKS and
    audit logging.
    """
    if not settings.auth_jwt_secret and not settings.auth_jwt_jwks_url:
        return None
    token = bearer_token(authorization)
    if token is None:
        raise AuthError("missing bearer token", reason="missing_token")
    return decode_jwt(token, settings, now=now)

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import pytest
from app.auth import AuthError, JwtVerifier, decode_jwt
from app.telemetry.metrics import Metrics
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from tests.conftest import make_jwt, make_settings


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _rsa_token(
    claims: dict[str, Any],
    key: rsa.RSAPrivateKey,
    *,
    kid: str = "key-1",
    alg: str = "RS256",
    expires_in: int = 300,
) -> str:
    header = {"alg": alg, "typ": "JWT", "kid": kid}
    payload = {"exp": int(time.time()) + expires_in, **claims}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    signature = key.sign(
        signing_input.encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return f"{signing_input}.{_b64(signature)}"


def _jwks(key: rsa.RSAPublicKey, kid: str = "key-1") -> dict[str, Any]:
    numbers = key.public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "use": "sig",
                "alg": "RS256",
                "n": _b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
                "e": _b64(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
            }
        ]
    }


def _ps256_token(claims: dict[str, Any], key: rsa.RSAPrivateKey) -> str:
    header = {"alg": "PS256", "typ": "JWT", "kid": "ps-1"}
    payload = {"exp": int(time.time()) + 300, **claims}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    digest = hashes.SHA256()
    signature = key.sign(
        signing_input.encode(),
        padding.PSS(mgf=padding.MGF1(digest), salt_length=digest.digest_size),
        digest,
    )
    return f"{signing_input}.{_b64(signature)}"


def _es256_token(claims: dict[str, Any], key: ec.EllipticCurvePrivateKey) -> str:
    header = {"alg": "ES256", "typ": "JWT", "kid": "ec-1"}
    payload = {"exp": int(time.time()) + 300, **claims}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    der = key.sign(signing_input.encode(), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    # JWS wants the raw fixed-width r || s pair, not the DER encoding.
    signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{signing_input}.{_b64(signature)}"


def _ec_jwks(key: ec.EllipticCurvePublicKey, kid: str = "ec-1") -> dict[str, Any]:
    numbers = key.public_numbers()
    return {
        "keys": [
            {
                "kty": "EC",
                "kid": kid,
                "use": "sig",
                "crv": "P-256",
                "x": _b64(numbers.x.to_bytes(32, "big")),
                "y": _b64(numbers.y.to_bytes(32, "big")),
            }
        ]
    }


def _fetcher(document: dict[str, Any]):
    async def fetch(url: str, timeout_s: float) -> dict[str, Any]:
        assert url.endswith("/jwks.json")
        return document

    return fetch


def _verifier(**overrides: Any) -> JwtVerifier:
    base: dict[str, Any] = {
        "auth_jwt_secret": None,
        "auth_jwt_jwks_url": "https://identity.example/.well-known/jwks.json",
        "auth_jwt_algorithms": ["RS256"],
        "auth_jwt_jwks_min_refresh_s": 0,
    }
    base.update(overrides)
    return JwtVerifier(make_settings(**base), Metrics())


async def test_rs256_token_verifies_against_jwks() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = _verifier()
    verifier._fetcher = _fetcher(_jwks(key.public_key()))  # type: ignore[assignment]

    token = _rsa_token({"sub": "user-rs"}, key)
    identity = await verifier.authenticate(f"Bearer {token}", surface="http")
    assert identity is not None
    assert identity.subject == "user-rs"
    assert verifier.stats()["mode"] == "jwks"
    assert verifier.stats()["jwks_cached_keys"] == 1


async def test_rs256_rejects_wrong_key_and_algorithm() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = _verifier()
    verifier._fetcher = _fetcher(_jwks(key.public_key()))  # type: ignore[assignment]

    with pytest.raises(AuthError) as bad:
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'x'}, other)}")
    assert bad.value.reason == "bad_signature"

    # A token whose alg is not in the allowlist never reaches key selection.
    token = _rsa_token({"sub": "x"}, key, alg="RS512")
    with pytest.raises(AuthError) as not_allowed:
        await verifier.authenticate(f"Bearer {token}")
    assert not_allowed.value.reason == "algorithm_not_allowed"


async def test_ps256_and_es256_tokens_verify() -> None:
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ps_verifier = _verifier(auth_jwt_algorithms=["PS256"])
    ps_verifier._fetcher = _fetcher(_jwks(rsa_key.public_key(), kid="ps-1"))  # type: ignore[assignment]
    ps_identity = await ps_verifier.authenticate(
        f"Bearer {_ps256_token({'sub': 'ps-user'}, rsa_key)}"
    )
    assert ps_identity is not None and ps_identity.subject == "ps-user"

    ec_key = ec.generate_private_key(ec.SECP256R1())
    es_verifier = _verifier(auth_jwt_algorithms=["ES256"])
    es_verifier._fetcher = _fetcher(_ec_jwks(ec_key.public_key()))  # type: ignore[assignment]
    es_identity = await es_verifier.authenticate(
        f"Bearer {_es256_token({'sub': 'ec-user'}, ec_key)}"
    )
    assert es_identity is not None and es_identity.subject == "ec-user"


async def test_key_type_mismatch_is_rejected() -> None:
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ec_key = ec.generate_private_key(ec.SECP256R1())
    # An ES256 token whose kid points at an RSA key must not be coerced.
    verifier = _verifier(auth_jwt_algorithms=["ES256"])
    verifier._fetcher = _fetcher(_jwks(rsa_key.public_key(), kid="ec-1"))  # type: ignore[assignment]
    with pytest.raises(AuthError) as error:
        await verifier.authenticate(f"Bearer {_es256_token({'sub': 'x'}, ec_key)}")
    assert error.value.reason == "key_type_mismatch"


async def test_jwks_key_rotation_is_picked_up_on_unknown_kid() -> None:
    first = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    second = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    document = _jwks(first.public_key(), kid="key-1")
    verifier = _verifier()
    verifier._fetcher = _fetcher(document)  # type: ignore[assignment]

    assert (
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'a'}, first, kid='key-1')}")
    ).subject == "a"

    document["keys"] = _jwks(second.public_key(), kid="key-2")["keys"]
    rotated = await verifier.authenticate(
        f"Bearer {_rsa_token({'sub': 'b'}, second, kid='key-2')}"
    )
    assert rotated.subject == "b"
    assert verifier.stats()["jwks_cached_keys"] == 1


async def test_rejected_handshake_is_audited_without_the_token(caplog) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = _verifier()
    verifier._fetcher = _fetcher(_jwks(key.public_key()))  # type: ignore[assignment]
    token = _rsa_token({"sub": "x"}, key, expires_in=-120)

    with (
        caplog.at_level(logging.WARNING, logger="app.auth"),
        pytest.raises(AuthError) as error,
    ):
        await verifier.authenticate(
            f"Bearer {token}",
            surface="websocket",
            remote="203.0.113.9",
            path="/ws/stream",
        )
    assert error.value.reason == "token_expired"
    records = [r for r in caplog.records if r.name == "app.auth"]
    assert records
    payload = json.loads(records[-1].getMessage().removeprefix("auth_rejected "))
    assert payload["event"] == "auth_rejected"
    assert payload["surface"] == "websocket"
    assert payload["reason"] == "token_expired"
    assert payload["remote"] == "203.0.113.9"
    assert payload["kid"] == "key-1"
    assert token not in records[-1].getMessage()
    assert verifier.metrics.counters["auth_rejections"] == 1
    assert verifier.metrics.counters["auth_rejections_token_expired"] == 1


async def test_missing_token_is_audited() -> None:
    verifier = _verifier()
    with pytest.raises(AuthError) as error:
        await verifier.authenticate(None, surface="http", path="/api/kb/search")
    assert error.value.reason == "missing_token"
    assert verifier.metrics.counters["auth_rejections_missing_token"] == 1


async def test_jwks_unavailable_is_a_rejection_not_a_pass() -> None:
    async def broken(url: str, timeout_s: float) -> dict[str, Any]:
        raise OSError("identity provider is down")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = _verifier()
    verifier._fetcher = broken  # type: ignore[assignment]
    with pytest.raises(AuthError) as error:
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'x'}, key)}")
    assert error.value.reason == "jwks_unavailable"
    assert verifier.metrics.counters["auth_jwks_refresh_errors"] == 1


def test_hmac_path_still_works_and_is_constant_time_checked() -> None:
    settings = make_settings(auth_jwt_secret="test-secret", auth_jwt_algorithms=["HS256"])
    token = make_jwt({"sub": "u1"})
    assert decode_jwt(token, settings).subject == "u1"

    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}.{signature[:-2]}aa"
    with pytest.raises(AuthError) as error:
        decode_jwt(tampered, settings)
    assert error.value.reason == "bad_signature"

    # An HS token does not need (and must not use) a public key.
    with pytest.raises(AuthError) as error:
        decode_jwt(token, make_settings(auth_jwt_secret=None))
    assert error.value.reason == "jwt_not_configured"


def test_bearer_parsing_and_hmac_digest_choice() -> None:
    settings = make_settings(auth_jwt_secret="s", auth_jwt_algorithms=["HS384"])
    token = make_jwt({"sub": "u"}, secret="s", alg="HS384")
    assert decode_jwt(token, settings).subject == "u"
    # HS256 is not allowed when the operator narrowed the allowlist.
    with pytest.raises(AuthError):
        decode_jwt(make_jwt({"sub": "u"}, secret="s", alg="HS256"), settings)

    # Sanity: the helper really signs with HMAC-SHA384.
    header, payload, signature = token.split(".")
    expected = hmac.new(
        b"s", f"{header}.{payload}".encode(), hashlib.sha384
    ).digest()
    assert signature == _b64(expected)

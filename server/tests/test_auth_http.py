"""JWKS over real HTTP, not a replaced fetcher.

The unit tests in `test_auth.py` swap `JwtVerifier._fetcher`, which proves the
key cache but not the fetch: status handling, JSON decoding, rotation over the
wire, and what a rejection looks like when the identity provider is down.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from app.auth import AuthError, JwtVerifier
from app.telemetry.metrics import Metrics
from cryptography.hazmat.primitives.asymmetric import rsa
from tests.conftest import make_settings
from tests.test_auth import _jwks, _rsa_token


class JwksSite:
    """A real JWKS endpoint on an ephemeral loopback port."""

    def __init__(self, document: dict[str, Any], status: int = 200) -> None:
        self.document = document
        self.status = status
        self.hits = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server names this
                site.hits += 1
                body = json.dumps(site.document).encode()
                self.send_response(site.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        return f"http://127.0.0.1:{server.server_address[1]}/jwks.json"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def _verifier(url: str, **overrides: Any) -> JwtVerifier:
    settings: dict[str, Any] = {
        "auth_jwt_secret": None,
        "auth_jwt_jwks_url": url,
        "auth_jwt_algorithms": ["RS256"],
        "auth_jwt_jwks_min_refresh_s": 0,
        "auth_jwt_jwks_cache_ttl_s": 30,
        "auth_audit_enabled": False,
    }
    settings.update(overrides)
    return JwtVerifier(make_settings(**settings), Metrics())


@pytest.fixture()
def site():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    endpoint = JwksSite(_jwks(key.public_key()))
    url = endpoint.start()
    try:
        yield endpoint, key, url
    finally:
        endpoint.stop()


async def test_rs256_verifies_over_real_http(site) -> None:
    endpoint, key, url = site
    verifier = _verifier(url)

    identity = await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'http-user'}, key)}")

    assert identity is not None
    assert identity.subject == "http-user"
    assert endpoint.hits == 1
    assert verifier.stats()["jwks_cached_keys"] == 1
    assert verifier.metrics.counters["auth_jwks_refreshed"] == 1


async def test_cached_keys_avoid_a_second_fetch(site) -> None:
    endpoint, key, url = site
    verifier = _verifier(url)
    token = f"Bearer {_rsa_token({'sub': 'cached'}, key)}"

    await verifier.authenticate(token)
    await verifier.authenticate(token)

    assert endpoint.hits == 1


async def test_rotation_is_picked_up_over_the_wire(site) -> None:
    endpoint, first, url = site
    verifier = _verifier(url)
    await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'before'}, first)}")

    second = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    endpoint.document = _jwks(second.public_key(), kid="key-2")

    identity = await verifier.authenticate(
        f"Bearer {_rsa_token({'sub': 'after'}, second, kid='key-2')}"
    )

    assert identity is not None
    assert identity.subject == "after"
    # The unknown kid forced one refresh; the document now holds the new key.
    assert endpoint.hits == 2
    assert verifier.stats()["jwks_cached_keys"] == 1


async def test_unknown_kid_refreshes_once_then_rejects(site) -> None:
    endpoint, key, url = site
    verifier = _verifier(url)
    await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'known'}, key)}")

    with pytest.raises(AuthError) as error:
        await verifier.authenticate(
            f"Bearer {_rsa_token({'sub': 'ghost'}, key, kid='missing')}"
        )

    assert error.value.reason == "unknown_key"
    assert endpoint.hits == 2


async def test_http_error_is_a_rejection_not_a_pass(site) -> None:
    endpoint, key, url = site
    endpoint.status = 500
    verifier = _verifier(url)

    with pytest.raises(AuthError) as error:
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'nope'}, key)}")

    assert error.value.reason == "jwks_unavailable"
    assert verifier.metrics.counters["auth_jwks_refresh_errors"] == 1
    assert verifier.stats()["jwks_cached_keys"] == 0


async def test_document_without_usable_keys_is_a_rejection(site) -> None:
    endpoint, key, url = site
    endpoint.document = {"keys": []}
    verifier = _verifier(url)

    with pytest.raises(AuthError) as error:
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'nope'}, key)}")

    assert error.value.reason == "jwks_empty"
    assert verifier.metrics.counters["auth_jwks_refresh_errors"] == 1


async def test_cached_keys_survive_an_outage_within_the_ttl(site) -> None:
    endpoint, key, url = site
    verifier = _verifier(url)
    token = f"Bearer {_rsa_token({'sub': 'cached'}, key)}"

    await verifier.authenticate(token)
    endpoint.stop()

    # The key cache is the fallback while the TTL holds: no fetch, no outage.
    identity = await verifier.authenticate(token)
    assert identity is not None
    assert identity.subject == "cached"


async def test_an_outage_after_the_ttl_rejects(site) -> None:
    endpoint, key, url = site
    verifier = _verifier(url, auth_jwt_jwks_cache_ttl_s=1)
    token = f"Bearer {_rsa_token({'sub': 'cached'}, key)}"

    await verifier.authenticate(token)
    endpoint.stop()
    await asyncio.sleep(1.05)

    # Past the TTL the verifier must refresh, and a failed refresh is a
    # rejection rather than a silent fallback. This is the bound the alert
    # description states.
    with pytest.raises(AuthError) as error:
        await verifier.authenticate(token)

    assert error.value.reason == "jwks_unavailable"


async def test_unreachable_endpoint_fails_fast() -> None:
    verifier = _verifier(
        "http://127.0.0.1:1/jwks.json",
        auth_jwt_jwks_timeout_s=0.5,
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with pytest.raises(AuthError) as error:
        await verifier.authenticate(f"Bearer {_rsa_token({'sub': 'nope'}, key)}")

    assert error.value.reason == "jwks_unavailable"

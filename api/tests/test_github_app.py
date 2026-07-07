"""
Tests for api/github_app.py — GitHub App JWT signing and installation tokens.

Covers:
- _build_app_jwt: signs a valid RS256 JWT with iss=app_id; raises when
  GITHUB_APP_ID/GITHUB_APP_PRIVATE_KEY are unconfigured
- GITHUB_APP_PRIVATE_KEY accepted either base64-encoded (the Azure-safe
  storage format) or as a raw PEM (local .env convenience)
- get_installation_token: mints via the GitHub API, caches, reuses the cache
  within the token's lifetime, refreshes once near expiry, falls back
  gracefully if GitHub's expires_at can't be parsed
- list_installations / list_installation_repos: pagination via the Link header
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timedelta, timezone

import httpx
import jwt as pyjwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import api.github_app as gh


@pytest.fixture(autouse=True)
def _default_installation_token():
    """
    Shadow conftest.py's autouse fixture of the same name (a fixture defined
    in a test module takes precedence over one from conftest.py). These
    tests exercise the real github_app functions directly, respx-mocking the
    underlying HTTP calls instead of mocking the functions themselves.
    """
    yield


@pytest.fixture()
def rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return key, pem


@pytest.fixture(autouse=True)
def clear_token_cache():
    gh._token_cache.clear()
    yield
    gh._token_cache.clear()


def _configure(monkeypatch, pem: str, app_id: str = "123456", b64: bool = True) -> None:
    monkeypatch.setenv("GITHUB_APP_ID", app_id)
    monkeypatch.setenv(
        "GITHUB_APP_PRIVATE_KEY",
        base64.b64encode(pem.encode()).decode() if b64 else pem,
    )


# ── _build_app_jwt ────────────────────────────────────────────────────────────

class TestBuildAppJwt:
    def test_raises_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("GITHUB_APP_ID", raising=False)
        monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
        with pytest.raises(RuntimeError):
            gh._build_app_jwt()

    def test_signs_valid_rs256_jwt_with_base64_key(self, monkeypatch, rsa_keypair):
        key, pem = rsa_keypair
        _configure(monkeypatch, pem)

        token = gh._build_app_jwt()
        decoded = pyjwt.decode(token, key.public_key(), algorithms=["RS256"])
        assert decoded["iss"] == "123456"
        assert decoded["exp"] > decoded["iat"]

    def test_accepts_raw_pem_not_just_base64(self, monkeypatch, rsa_keypair):
        key, pem = rsa_keypair
        _configure(monkeypatch, pem, b64=False)

        token = gh._build_app_jwt()
        decoded = pyjwt.decode(token, key.public_key(), algorithms=["RS256"])
        assert decoded["iss"] == "123456"


# ── get_installation_token ────────────────────────────────────────────────────

class TestGetInstallationToken:
    @pytest.mark.asyncio
    async def test_mints_token_via_api(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with respx.mock() as router:
            router.post("https://api.github.com/app/installations/42/access_tokens").mock(
                return_value=httpx.Response(201, json={"token": "ghs_abc123", "expires_at": expires})
            )
            token = await gh.get_installation_token(42)

        assert token == "ghs_abc123"

    @pytest.mark.asyncio
    async def test_reuses_cached_token_without_refetching(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with respx.mock() as router:
            route = router.post("https://api.github.com/app/installations/42/access_tokens").mock(
                return_value=httpx.Response(201, json={"token": "ghs_first", "expires_at": expires})
            )
            first = await gh.get_installation_token(42)
            second = await gh.get_installation_token(42)

        assert first == second == "ghs_first"
        assert route.call_count == 1  # second call served from cache, no re-mint

    @pytest.mark.asyncio
    async def test_refreshes_when_cached_token_near_expiry(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        # Seed the cache with a token inside the 5-minute refresh margin.
        gh._token_cache[42] = ("ghs_stale", time.time() + 60)

        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with respx.mock() as router:
            router.post("https://api.github.com/app/installations/42/access_tokens").mock(
                return_value=httpx.Response(201, json={"token": "ghs_fresh", "expires_at": expires})
            )
            token = await gh.get_installation_token(42)

        assert token == "ghs_fresh"

    @pytest.mark.asyncio
    async def test_expiry_parse_failure_falls_back_gracefully(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        with respx.mock() as router:
            router.post("https://api.github.com/app/installations/42/access_tokens").mock(
                return_value=httpx.Response(201, json={"token": "ghs_x", "expires_at": "not-a-date"})
            )
            token = await gh.get_installation_token(42)

        assert token == "ghs_x"
        _, cached_expiry = gh._token_cache[42]
        assert cached_expiry > time.time()  # fell back to ~1hr from now, not garbage


# ── list_installations / list_installation_repos ─────────────────────────────

class TestListInstallations:
    @pytest.mark.asyncio
    async def test_paginates_via_link_header(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        # A single route: respx matches a query-less pattern against any query
        # string on that path, so a page-1 route and a page-2 route registered
        # separately would both match every call (the first-registered one
        # always wins) — use side_effect to return successive pages from one
        # route instead, which is also what real pagination actually needs to
        # assert (same path, different content per call).
        page2_url = "https://api.github.com/app/installations?per_page=100&page=2"
        with respx.mock() as router:
            router.get("https://api.github.com/app/installations").mock(
                side_effect=[
                    httpx.Response(
                        200, json=[{"id": 1}],
                        headers={"Link": f'<{page2_url}>; rel="next"'},
                    ),
                    httpx.Response(200, json=[{"id": 2}]),  # no Link header -> loop stops
                ]
            )

            installations = await gh.list_installations()

        assert [i["id"] for i in installations] == [1, 2]


class TestListInstallationRepos:
    @pytest.mark.asyncio
    async def test_returns_full_names(self, monkeypatch, rsa_keypair):
        _, pem = rsa_keypair
        _configure(monkeypatch, pem)

        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with respx.mock() as router:
            router.post("https://api.github.com/app/installations/42/access_tokens").mock(
                return_value=httpx.Response(201, json={"token": "ghs_x", "expires_at": expires})
            )
            router.get("https://api.github.com/installation/repositories").mock(
                return_value=httpx.Response(
                    200,
                    json={"repositories": [{"full_name": "owner/repo-a"}, {"full_name": "owner/repo-b"}]},
                )
            )
            repos = await gh.list_installation_repos(42)

        assert repos == ["owner/repo-a", "owner/repo-b"]

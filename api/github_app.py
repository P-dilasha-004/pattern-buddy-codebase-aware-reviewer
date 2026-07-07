"""
GitHub App authentication — JWT signing and per-installation access tokens.

Replaces the old static PAT: every GitHub API call is authenticated as a
specific installation, auto-scoped by GitHub to whatever repos that
installation was granted at install time. No manual token scoping on
github.com is required for a new repo — installing the App *is* the grant.

Public API
----------
get_installation_token(installation_id) -> str   (cached, auto-refreshing)
list_installations() -> list[dict]
list_installation_repos(installation_id) -> list[str]
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
import time

import httpx
import jwt

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# GitHub's hard cap on App JWT lifetime is 10 minutes; leave a safety margin.
_JWT_TTL_S: int = 9 * 60
# Installation tokens live 1hr; refresh a bit early rather than racing expiry
# mid-request.
_TOKEN_REFRESH_MARGIN_S: int = 5 * 60

_HTTP_TIMEOUT_S: float = 30.0


def _load_private_key(raw: str) -> str:
    """
    GITHUB_APP_PRIVATE_KEY holds the App's PEM private key, base64-encoded —
    Azure App Settings mangle literal newlines in multiline env values, so
    the key is stored as one base64 blob and decoded here. A raw (already
    PEM-formatted, with real newlines) value is accepted too, for local .env
    convenience.
    """
    if not raw:
        return ""
    if "BEGIN" in raw:
        return raw
    try:
        return base64.b64decode(raw).decode("utf-8")
    except Exception:
        return raw


def _app_id() -> str:
    return os.environ.get("GITHUB_APP_ID", "")


def _private_key() -> str:
    return _load_private_key(os.environ.get("GITHUB_APP_PRIVATE_KEY", ""))


# installation_id -> (token, expires_at_epoch_s)
_token_cache: dict[int, tuple[str, float]] = {}


def _build_app_jwt() -> str:
    """Short-lived JWT identifying the App itself (not any installation)."""
    app_id = _app_id()
    private_key = _private_key()
    if not app_id or not private_key:
        raise RuntimeError(
            "GITHUB_APP_ID / GITHUB_APP_PRIVATE_KEY not configured — "
            "cannot authenticate as the GitHub App"
        )
    now = int(time.time())
    payload = {
        "iat": now - 60,  # allow for clock drift between this host and GitHub
        "exp": now + _JWT_TTL_S,
        "iss": app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def _parse_expiry(expires_at: str) -> float:
    try:
        return (
            datetime.datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
    except (ValueError, TypeError):
        logger.warning("installation_token_expiry_parse_failed raw=%s", expires_at)
        return time.time() + 3600  # GitHub's documented token lifetime


async def get_installation_token(
    installation_id: int, http_timeout: float = _HTTP_TIMEOUT_S
) -> str:
    """
    Return a valid installation access token, minting (or refreshing) one via
    the GitHub API when the cached copy is missing or near expiry.

    Cached in-process (module-level dict) — safe under the deployment's
    single gunicorn worker, same assumption the existing concurrency guards
    in api/main.py already rely on.
    """
    cached = _token_cache.get(installation_id)
    if cached and cached[1] - time.time() > _TOKEN_REFRESH_MARGIN_S:
        return cached[0]

    app_jwt = _build_app_jwt()
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(
            f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    token = data["token"]
    expires_at = _parse_expiry(data.get("expires_at", ""))
    _token_cache[installation_id] = (token, expires_at)
    logger.info("installation_token_minted installation_id=%s", installation_id)
    return token


async def list_installations(http_timeout: float = _HTTP_TIMEOUT_S) -> list[dict]:
    """
    All accounts that have installed this App. JWT-authenticated (App-level,
    not installation-level) — used to resolve repo_id -> installation_id when
    no live webhook context has already supplied it (e.g. a manual
    /admin/reindex-all trigger).
    """
    app_jwt = _build_app_jwt()
    installations: list[dict] = []
    url: str | None = f"{GITHUB_API_BASE}/app/installations"
    params: dict | None = {"per_page": 100}

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        while url:
            resp = await client.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            installations.extend(resp.json())
            url = resp.links.get("next", {}).get("url")
            params = None  # the next URL already carries its own query params

    return installations


async def list_installation_repos(
    installation_id: int, http_timeout: float = _HTTP_TIMEOUT_S
) -> list[str]:
    """repo_id ("owner/name") strings accessible to a given installation."""
    token = await get_installation_token(installation_id, http_timeout=http_timeout)
    repos: list[str] = []
    url: str | None = f"{GITHUB_API_BASE}/installation/repositories"
    params: dict | None = {"per_page": 100}

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        while url:
            resp = await client.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            repos.extend(r["full_name"] for r in data.get("repositories", []))
            url = resp.links.get("next", {}).get("url")
            params = None

    return repos

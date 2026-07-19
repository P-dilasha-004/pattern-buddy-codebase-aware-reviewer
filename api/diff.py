"""
Phase 2, Task 2.2 — Diff extraction and concurrent file fetching.

Public API
----------
extract_and_fetch(payload, pat) -> FetchResult

FetchResult.fallback is True when the PR touches >50 ingestible files.
In that case .files is empty and downstream tasks must use the macro-summary
path (commit message + file names only, no Qdrant retrieval).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GITHUB_API_BASE = "https://api.github.com"
FETCH_TIMEOUT_S: float = 30.0          # per-request; hung GitHub blob never blocks forever
MAX_FILES_THRESHOLD: int = 50          # >50 ingestible files → macro-summary fallback

# Extensions the ingestion pipeline can process (mirrors pipeline.py routing tables)
_INGESTIBLE_EXTS: frozenset[str] = frozenset({
    ".py",                              # Track A
    ".md", ".txt",                      # Track B
    ".toml", ".json", ".yaml", ".yml",  # Track C
})

# Filenames always skipped regardless of extension (mirrors pipeline.py SKIP_FILENAMES)
_SKIP_NAMES: frozenset[str] = frozenset({
    "poetry.lock", "package-lock.json", "yarn.lock", "uv.lock",
    "Pipfile.lock", "composer.lock", "Gemfile.lock", "cargo.lock",
    "packages.lock.json",
})


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    files: dict[str, str] = field(default_factory=dict)  # path -> source text
    fallback: bool = False                                # True → >50 files, use macro path
    failed_paths: list[str] = field(default_factory=list) # paths whose fetch failed


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_ingestible(path: str) -> bool:
    p = Path(path)
    if p.name.lower() in _SKIP_NAMES:
        return False
    if p.suffix.lower() == ".lock":
        return False
    return p.suffix.lower() in _INGESTIBLE_EXTS


def _fetchable(changed_files: list[dict]) -> list[dict]:
    """Return files that are ingestible and have content (not deleted)."""
    return [
        f for f in changed_files
        if f.get("status") != "removed"
        and _is_ingestible(f.get("path", ""))
        and f.get("path")
    ]


async def _fetch_one(
    client: httpx.AsyncClient, entry: dict, repo_id: str, ref: str, pat: str
) -> tuple[str, str]:
    """
    Fetch a single file's raw content via the GitHub Contents API.

    We deliberately do NOT use the pulls/files `raw_url`: GitHub returns it with
    percent-encoded slashes (e.g. `src%2Fbackend%2Ffile.py`) that 404 for nested
    paths, and it points at the github.com/raw redirect to raw.githubusercontent.com
    where httpx drops the Authorization header cross-host — so it silently 404s on
    private repos. The Contents API with the raw media type takes the path in the
    URL, keeps auth on a single host, and returns file bytes directly.

    PAT is injected on every request. Raises on non-2xx — caller handles via
    return_exceptions.
    """
    path = entry["path"]
    url = f"{GITHUB_API_BASE}/repos/{repo_id}/contents/{path}"
    resp = await client.get(
        url,
        params={"ref": ref},
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github.raw",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    resp.raise_for_status()
    return path, resp.text


# ── Public entry point ────────────────────────────────────────────────────────

async def extract_and_fetch(
    payload: dict,
    pat: str,
    http_timeout: float = FETCH_TIMEOUT_S,
) -> FetchResult:
    """
    Parse the PR payload, apply the 50-file defensive threshold, then
    concurrently fetch all ingestible file blobs from GitHub.

    Args:
        payload:      Webhook payload from POST /review.
        pat:          GitHub Personal Access Token for authenticated fetches.
        http_timeout: Per-request timeout in seconds (default 30 s).

    Returns:
        FetchResult with .files (path→content), .fallback, and .failed_paths.
    """
    changed = payload.get("changed_files", [])
    to_fetch = _fetchable(changed)

    # ── Defensive threshold ───────────────────────────────────────────────────
    if len(to_fetch) > MAX_FILES_THRESHOLD:
        logger.warning(
            "large_pr_fallback repo=%s pr=%s ingestible_files=%d threshold=%d",
            payload.get("repo_id"), payload.get("pr_number"),
            len(to_fetch), MAX_FILES_THRESHOLD,
        )
        return FetchResult(fallback=True)

    if not to_fetch:
        return FetchResult()

    repo_id = payload.get("repo_id", "")
    # /review sends head_sha; /sync sends commit_hash. Fall back to HEAD.
    ref = payload.get("head_sha") or payload.get("commit_hash") or "HEAD"

    # ── Concurrent fetch ──────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=http_timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *[_fetch_one(client, f, repo_id, ref, pat) for f in to_fetch],
            return_exceptions=True,
        )

    files: dict[str, str] = {}
    failed: list[str] = []

    for entry, result in zip(to_fetch, results):
        if isinstance(result, Exception):
            logger.warning(
                "fetch_failed repo=%s path=%s error=%s",
                payload.get("repo_id"), entry["path"], result,
            )
            failed.append(entry["path"])
        else:
            path, content = result
            files[path] = content

    logger.info(
        "fetch_complete repo=%s pr=%s fetched=%d failed=%d",
        payload.get("repo_id"), payload.get("pr_number"),
        len(files), len(failed),
    )
    return FetchResult(files=files, failed_paths=failed)

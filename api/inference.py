"""
Phase 2, Task 2.4 — Inference handoff and GitHub callback.

Public API
----------
call_inference(prompt, phi_endpoint, phi_key, http_timeout) -> str
post_github_comment(repo_id, pr_number, body, pat, http_timeout) -> None

The Azure AI Foundry endpoint uses the OpenAI-compatible chat completions API.
GitHub comment posting uses the Issues Comments endpoint (PRs share the
issues namespace in the GitHub API).
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_PHI_CONTEXT_WINDOW: int = 8_192   # Phi-4-mini-reasoning hard limit

from api.retrieval import PROMPT_TOKEN_CAP  # noqa: E402 — avoids circular at call time

# Generation budget. This is a CEILING, not a target — the model stops at EOS when
# done, so a larger cap doesn't slow simple reviews; it just prevents the model
# from being cut off mid-answer. Phi-4-mini-reasoning spends a big chunk of its
# output inside a <think> block before writing the review, so a tight cap (the
# old 256) truncated it while it was still reasoning: no closing </think>, no
# actual verdict. 1500 leaves room to finish reasoning AND produce the review.
# Must satisfy PROMPT_TOKEN_CAP + MAX_NEW_TOKENS <= context window (6000 + 1500 < 8192).
MAX_NEW_TOKENS: int = 1_500

INFERENCE_TIMEOUT_S: float = 180.0   # Phi-4-mini-reasoning can be slow; cap at 3 min
GITHUB_TIMEOUT_S:    float = 30.0

# The Azure AI Foundry serverless endpoint has a tight per-minute rate limit.
# The review runs in a background task (GitHub already got its 202), so we can
# afford to wait out a 429/503 rather than posting the error comment. Backoff
# honours the Retry-After header when present, else exponential (capped).
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 503})
MAX_INFERENCE_RETRIES: int = 5
_BACKOFF_BASE_S: float = 4.0
_BACKOFF_CAP_S:  float = 60.0

GITHUB_API_BASE = "https://api.github.com"

FALLBACK_COMMENT = (
    "_Pattern Buddy encountered an error during review. "
    "Please check the App Service logs for details._"
)


# ── Inference ─────────────────────────────────────────────────────────────────

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)   # complete blocks
_THINK_OPEN_RE   = re.compile(r"<think>.*\Z", re.DOTALL)          # unclosed (truncated)

_TRUNCATED_MSG = (
    "_Pattern Buddy could not complete the review — the model ran out of its "
    "generation budget while reasoning and did not produce a verdict. This "
    "usually clears on a re-run; if it persists the diff may be too large._"
)


def _strip_reasoning(raw: str) -> str:
    """
    Remove the reasoning model's <think> content so only the review reaches the PR.

    Handles both complete <think>…</think> blocks and the truncated case where the
    model was cut off mid-reasoning and never emitted a closing </think> (the old
    regex missed that and posted the raw monologue). Any stray tags are cleared too.
    """
    cleaned = _THINK_CLOSED_RE.sub("", raw)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)          # drop an unclosed trailing block
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: honour Retry-After, else capped exp backoff."""
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), _BACKOFF_CAP_S)
        except ValueError:
            pass  # HTTP-date form is rare here; fall through to backoff
    return min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_CAP_S)


async def call_inference(
    prompt: str,
    phi_endpoint: str,
    phi_key: str,
    phi_model: str = "Phi-4-mini-reasoning",
    http_timeout: float = INFERENCE_TIMEOUT_S,
) -> str:
    """
    POST the assembled prompt to the Azure AI Foundry Phi-4-mini-reasoning
    serverless endpoint and return the generated review text.

    The endpoint uses the OpenAI-compatible chat completions API format.
    Retries on 429/503 (rate limit) with backoff; raises httpx.HTTPStatusError
    on other non-2xx, on the final retry, or httpx.TimeoutException on timeout.
    """
    headers = {
        "Authorization": f"Bearer {phi_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": phi_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
    }

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        for attempt in range(MAX_INFERENCE_RETRIES + 1):
            resp = await client.post(phi_endpoint, json=body, headers=headers)
            if resp.status_code in _RETRYABLE_STATUS and attempt < MAX_INFERENCE_RETRIES:
                delay = _retry_delay(resp, attempt)
                logger.warning(
                    "inference_rate_limited status=%s attempt=%d/%d retry_in=%.1fs",
                    resp.status_code, attempt + 1, MAX_INFERENCE_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()   # 2xx passes; other errors (or final 429) raise
            break

    data = resp.json()
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected inference response shape: {data!r}") from exc

    review = _strip_reasoning(raw)
    # If the whole response was reasoning (model truncated before a verdict), the
    # stripped result is empty — post a clear message rather than a blank comment
    # or raw <think> text.
    return review or _TRUNCATED_MSG


# ── GitHub comment ────────────────────────────────────────────────────────────

def _comment_url(repo_id: str, pr_number: int) -> str:
    """
    PRs share the GitHub Issues namespace.
    POST /repos/{owner}/{repo}/issues/{number}/comments
    """
    return f"{GITHUB_API_BASE}/repos/{repo_id}/issues/{pr_number}/comments"


async def post_github_comment(
    repo_id: str,
    pr_number: int,
    body: str,
    pat: str,
    http_timeout: float = GITHUB_TIMEOUT_S,
) -> None:
    """
    Post a Markdown comment to the GitHub pull request.
    Raises httpx.HTTPStatusError on non-2xx.
    """
    url = _comment_url(repo_id, pr_number)
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(url, json={"body": body}, headers=headers)
        resp.raise_for_status()

    logger.info(
        "comment_posted repo=%s pr=%s chars=%d",
        repo_id, pr_number, len(body),
    )

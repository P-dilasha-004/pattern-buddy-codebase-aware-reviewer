"""
Phase 2, Task 2.3 — Retrieval and prompt assembly.

Public API
----------
retrieve_and_assemble(files, payload, qdrant_client, embedder) -> PromptResult
build_macro_prompt(payload) -> str

PromptResult.prompt is the fully assembled string ready for the inference
endpoint.  PromptResult.truncated is True when chunks were dropped to fit
the 7,000-token cap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION      = "global_codebase_memory"
TOP_K           = 5
# Empirically tuned for nomic-embed-text-v1.5 on code. planning.md's 0.75 was too
# aggressive: relevant repo chunks for a code diff score ~0.62–0.72 with this model,
# so 0.75 rejected ALL of them → 0 chunks retrieved → reviews ran diff-only (not
# codebase-aware) despite the repo being ingested. Measured noise floor (unrelated
# text vs. this repo) is ~0.47–0.50, so 0.60 sits in the gap: keeps relevant
# context, filters noise.
SCORE_THRESHOLD = 0.60
PROMPT_TOKEN_CAP = 6_000   # input cap; leaves 1,500 for reasoning+generation within the 8,192 window (see inference.MAX_NEW_TOKENS)

_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENC.encode(text))


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    prompt: str
    truncated: bool = False          # True when chunks were dropped for the cap
    chunks_used: int = 0             # how many Qdrant chunks made it into prompt
    chunks_retrieved: int = 0        # how many Qdrant returned before truncation


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_BLOCK = """\
=========================
SYSTEM
=========================

You are a Senior Principal Engineer responsible for enforcing
repository standards and architectural consistency.

Your job is NOT to approve code.

Your job is to identify:
- Rule violations
- Missing tests
- Security risks
- Architectural inconsistencies
- Documentation violations

Only enforce rules appearing in retrieved repository context.

Do not hallucinate rules.

Output valid Markdown.

If no rule applies, state: "No repository rule violation detected.\""""

_TASK_BLOCK = """\
=========================
TASK
=========================

Review the pull request.

Identify:
1. Violated repository rules
2. Security concerns
3. Missing tests
4. Documentation gaps

For every finding:

- Severity
- Explanation
- Supporting repository rule
- Suggested fix

If no violation exists, explicitly state that.

=========================
OUTPUT FORMAT
=========================

# Review Summary

## Finding 1

Severity: High

Rule Source:
<file>

Issue:
<description>

Suggested Fix:
<description>"""


def _diff_block(files: dict[str, str]) -> str:
    if not files:
        return "(no file content fetched)"
    parts = []
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _context_block(chunks: list[str]) -> str:
    if not chunks:
        return "(no repository context retrieved)"
    return "\n\n---\n\n".join(chunks)


def _assemble(context: str, diff: str) -> str:
    return (
        f"{_SYSTEM_BLOCK}\n\n"
        f"=========================\n"
        f"REPOSITORY CONTEXT\n"
        f"=========================\n\n"
        f"{context}\n\n"
        f"=========================\n"
        f"PULL REQUEST DIFF\n"
        f"=========================\n\n"
        f"{diff}\n\n"
        f"{_TASK_BLOCK}"
    )


# ── Macro-summary prompt (>50 files fallback) ─────────────────────────────────

def build_macro_prompt(payload: dict) -> str:
    """
    Lightweight prompt used when a PR touches >50 files.
    Skips Qdrant retrieval entirely; uses only commit metadata and file names.
    """
    repo_id      = payload.get("repo_id", "unknown")
    pr_number    = payload.get("pr_number", "?")
    commit_msg   = payload.get("commit_message", "(no commit message provided)")
    changed      = payload.get("changed_files", [])
    file_names   = [f.get("path", "") if isinstance(f, dict) else str(f) for f in changed]
    file_list    = "\n".join(f"- {p}" for p in file_names) or "(none)"

    return (
        f"{_SYSTEM_BLOCK}\n\n"
        f"=========================\n"
        f"MACRO-LEVEL SUMMARY\n"
        f"=========================\n\n"
        f"Repository: {repo_id}\n"
        f"PR: #{pr_number}\n"
        f"Commit message: {commit_msg}\n\n"
        f"This PR modifies {len(file_names)} files — too many for line-by-line analysis.\n\n"
        f"Changed files:\n{file_list}\n\n"
        f"{_TASK_BLOCK}"
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _embed_files(files: dict[str, str], embedder) -> list[float]:
    """
    Embed the concatenated diff content to use as the Qdrant query vector.
    Concatenates all file content; fastembed truncates at 8192 tokens internally.
    """
    combined = "\n\n".join(f"# {path}\n{content}" for path, content in files.items())
    vectors = list(embedder.embed([combined]))
    return vectors[0].tolist()


def _query_qdrant(
    client: QdrantClient,
    repo_id: str,
    vector: list[float],
) -> list[tuple[str, float]]:
    """
    Query Qdrant with mandatory repo_id filter.
    Returns list of (content, score) sorted by score descending.
    """
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=repo_id),
                )
            ]
        ),
        limit=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        with_payload=True,
    )
    return [
        (point.payload.get("content", ""), point.score)
        for point in results.points
    ]


# ── Truncation ────────────────────────────────────────────────────────────────

def _truncate_to_cap(
    chunks_with_scores: list[tuple[str, float]],
    diff: str,
) -> tuple[list[str], bool]:
    """
    Drop lowest-scoring chunks until the assembled prompt fits within
    PROMPT_TOKEN_CAP.  System and Diff blocks are always preserved.
    Returns (kept_chunks, was_truncated).
    """
    # Tokens consumed by everything except the context block
    base_tokens = _count(_assemble("", diff))

    kept = list(chunks_with_scores)   # highest score first (Qdrant order)
    truncated = False

    while kept:
        context = _context_block([c for c, _ in kept])
        total = _count(_assemble(context, diff))
        if total <= PROMPT_TOKEN_CAP:
            break
        # Drop the lowest-scoring chunk (last in the list)
        kept.pop()
        truncated = True
    else:
        # kept is empty — even zero chunks might exceed cap (huge diff)
        truncated = bool(chunks_with_scores)

    return [c for c, _ in kept], truncated


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve_and_assemble(
    files: dict[str, str],
    payload: dict,
    qdrant_client: QdrantClient,
    embedder,
    on_stage: Callable[[str, float, dict], None] | None = None,
) -> PromptResult:
    """
    Embed the fetched diff, query Qdrant, apply truncation, and return the
    assembled prompt string.

    Args:
        files:         path→content dict from Task 2.2.
        payload:       original webhook payload (for repo_id).
        qdrant_client: connected Qdrant client.
        embedder:      fastembed TextEmbedding instance (already loaded).
        on_stage:      optional callback(stage_name, latency_ms, metadata) invoked
                       after "embedding" and after "qdrant_retrieval" complete —
                       used by Task 5.1 to emit per-stage trace logs without this
                       function needing to know how logging is implemented.
    """
    repo_id = payload.get("repo_id", "")
    diff    = _diff_block(files)

    # Embed diff → query vector
    t0 = time.monotonic()
    vector = _embed_files(files, embedder)
    if on_stage:
        on_stage("embedding", (time.monotonic() - t0) * 1000, {"file_count": len(files)})

    # Retrieve from Qdrant with mandatory repo_id filter
    t0 = time.monotonic()
    chunks_with_scores = _query_qdrant(qdrant_client, repo_id, vector)
    retrieved_count    = len(chunks_with_scores)
    if on_stage:
        on_stage("qdrant_retrieval", (time.monotonic() - t0) * 1000, {"chunks_retrieved": retrieved_count})

    logger.info(
        "qdrant_retrieved repo=%s pr=%s chunks=%d",
        repo_id, payload.get("pr_number"), retrieved_count,
    )

    # Truncate to cap, preserving System + Diff blocks
    kept_chunks, truncated = _truncate_to_cap(chunks_with_scores, diff)

    if truncated:
        logger.warning(
            "prompt_truncated repo=%s pr=%s kept=%d dropped=%d",
            repo_id, payload.get("pr_number"),
            len(kept_chunks), retrieved_count - len(kept_chunks),
        )

    context = _context_block(kept_chunks)
    prompt  = _assemble(context, diff)

    return PromptResult(
        prompt=prompt,
        truncated=truncated,
        chunks_used=len(kept_chunks),
        chunks_retrieved=retrieved_count,
    )

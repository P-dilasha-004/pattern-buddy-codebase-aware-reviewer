"""
Phase 2 — Backend API Orchestrator.

Entry point for the Azure App Service.
Run locally: uvicorn api.main:app --reload
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client import models as qdrant_models

from api.diff import extract_and_fetch
from api.retrieval import retrieve_and_assemble, build_macro_prompt
from api.inference import call_inference, post_github_comment, FALLBACK_COMMENT
from ingestion.pipeline import (
    QDRANT_COLLECTION,
    ingest_repository,
    _walk_repo,
    _delete_file_vectors,
    _chunk_file,
    _embed_chunks,
    _upsert_batch,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Pattern Buddy", version="2.0.0")

# ── Security ──────────────────────────────────────────────────────────────────

_WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")
_PAT:            str = os.environ.get("PAT", "")
_PHI_ENDPOINT:   str = os.environ.get("PHI_ENDPOINT", "")
_PHI_KEY:        str = os.environ.get("PHI_KEY", "")
_PHI_DEPLOYMENT: str = os.environ.get("PHI_DEPLOYMENT_NAME", "Phi-4-mini-reasoning")

# repo_ids with an in-flight /admin/reindex. Overlapping reindexes for the same
# repo interleave delete+upsert and inflate the vector count with duplicates, so
# a second trigger is rejected while one is running. Sufficient with the single
# gunicorn worker (-w 1) — the set lives in that one process's event loop, and
# the check-and-add in the endpoint is atomic (no await between them).
_reindex_in_progress: set[str] = set()

# repo_ids with an in-flight /review. The Phi endpoint is rate-limited to 1
# request / 60s, and each review (plus its retries) consumes that budget, so
# overlapping reviews for a repo — from rapid open/close/reopen/push on a PR —
# starve each other and all fail with the fallback comment. Serialize reviews
# per repo: a second trigger while one is running is rejected (workflow-level
# concurrency already cancels superseded runs, so we don't lose the latest).
_review_in_progress: set[str] = set()

def _get_qdrant() -> "QdrantClient":
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", ""),
        api_key=os.environ.get("QDRANT_API_KEY", ""),
    )

# Cap ONNX Runtime intra-op threads. On the single-worker (-w 1) App Service,
# an uncapped embedder pegs every core during ingestion, starves the asyncio
# event-loop thread that answers Azure's container health pings, and the worker
# gets SIGKILL'd ~30s in (misreported by gunicorn as "out of memory" — measured
# peak was <1GB). Leaving a core free keeps the loop responsive. Configurable so
# it can be tuned per plan (B3 = 4 vCores → default 2).
_EMBED_THREADS: int = int(os.environ.get("EMBED_THREADS", "2"))

# Loaded once at startup — reloading the ONNX model per request causes OOM on App Service.
def _load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", threads=_EMBED_THREADS)

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _load_embedder()
    return _embedder


# ── Observability (Task 5.1 — trace ID & structured logging) ────────────────

LOGS_DIR = Path(os.environ.get("PATTERN_BUDDY_LOGS_DIR", "logs"))


def _new_trace_id(repo_id: str, pr_number: int | str) -> str:
    """trace_id = pr_{repo}_{pr_number}_{timestamp} — planning.md section 9.2."""
    safe_repo = (repo_id or "unknown").replace("/", "-")
    return f"pr_{safe_repo}_{pr_number}_{int(time.time())}"


def _emit_stage_log(
    trace_id: str,
    stage: str,
    status: str,
    latency_ms: float,
    metadata: dict | None = None,
) -> None:
    """
    Emit one structured JSON log entry per planning.md section 9.6, and
    append it to logs/<stage>.json (JSON Lines) per section 9.7's per-stage
    file layout. Always logs to stdout first so the trace survives even if
    the local disk write fails or App Service's filesystem is ephemeral.
    """
    entry = {
        "trace_id": trace_id,
        "stage": stage,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "metadata": metadata or {},
    }
    logger.info("stage_log %s", json.dumps(entry))
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / f"{stage}.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("stage_log_write_failed stage=%s error=%s", stage, exc)


@contextmanager
def _stage_timer(trace_id: str, stage: str):
    """
    Times a pipeline stage and emits a structured log on exit — "success"
    with whatever metadata the caller attached, or "failed" if the block
    raised. Re-raises so the outer pipeline's own error handling still runs.

    Usage:
        with _stage_timer(trace_id, "diff_fetch") as ctx:
            result = do_work()
            ctx["metadata"] = {"count": len(result)}
    """
    ctx: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        yield ctx
    except Exception:
        ctx["latency_ms"] = (time.monotonic() - t0) * 1000
        _emit_stage_log(trace_id, stage, "failed", ctx["latency_ms"], ctx.get("metadata"))
        raise
    else:
        ctx["latency_ms"] = (time.monotonic() - t0) * 1000
        _emit_stage_log(trace_id, stage, "success", ctx["latency_ms"], ctx.get("metadata"))


# ── Observability (Task 5.2 — system metrics collection) ────────────────────

def _current_ram_mb() -> float:
    """
    Peak resident set size for this worker process, in MB — the exact signal
    Task 0.1 cared about (ONNX embedder OOM risk on App Service's 3.5 GB cap).
    ru_maxrss units differ by platform: KB on Linux (App Service), bytes on
    macOS (local dev) — normalize both to MB.
    """
    import resource
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _emit_system_metrics(
    trace_id: str,
    status: str,
    end_to_end_latency_ms: float,
    inference_latency_ms: float | None,
) -> None:
    """
    Emit one system-metrics record per review — planning.md section 9.5's
    end_to_end_latency_ms and (ACI-era naming) inference_latency_ms, plus
    current worker RAM utilization. Appends to logs/system_metrics.json.
    """
    entry = {
        "trace_id": trace_id,
        "status": status,
        "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
        "inference_latency_ms": round(inference_latency_ms, 2) if inference_latency_ms is not None else None,
        "ram_mb": round(_current_ram_mb(), 2),
    }
    logger.info("system_metrics %s", json.dumps(entry))
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / "system_metrics.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("system_metrics_write_failed error=%s", exc)


def _verify_signature(body: bytes, signature_header: str | None) -> None:
    """
    Validate the GitHub HMAC-SHA256 webhook signature.

    GitHub sends: x-hub-signature-256: sha256=<hex>
    Raises HTTP 401 if missing or invalid.
    """
    if not _WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET not configured")

    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing or malformed signature")

    expected = "sha256=" + hmac.new(
        _WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Signature mismatch")


# ── Background task (stub — filled in by Task 2.2) ───────────────────────────

async def _process_review(payload: dict[str, Any]) -> None:
    """
    Full review pipeline — runs in the background after 202 is sent to GitHub.

    On any exception, posts a fallback PR comment so the developer is never
    silently left without feedback (closes the gap described in Task 2.1 review).

    Task 5.1: every stage (diff_fetch, embedding, qdrant_retrieval,
    inference_handoff) emits a structured trace log under a shared trace_id
    so a review's full pipeline can be reconstructed after the fact.

    Task 5.2: end_to_end_latency_ms, inference_latency_ms, and current
    worker RAM utilization are recorded to logs/system_metrics.json
    regardless of success or failure.
    """
    import asyncio

    repo_id   = payload.get("repo_id", "")
    pr_number = payload.get("pr_number", 0)
    trace_id  = payload.get("trace_id") or _new_trace_id(repo_id, pr_number)

    logger.info(
        "review_started repo=%s pr=%s head=%s trace_id=%s",
        repo_id, pr_number, payload.get("head_sha"), trace_id,
    )

    t_pipeline_start = time.monotonic()
    inference_latency_ms: float | None = None
    status = "success"

    try:
        # Task 2.2 — diff extraction + concurrent file fetch
        with _stage_timer(trace_id, "diff_fetch") as ctx:
            fetch_result = await extract_and_fetch(payload, pat=_PAT)
            ctx["metadata"] = {
                "files_fetched": len(fetch_result.files),
                "fallback": fetch_result.fallback,
                "failed_paths": len(fetch_result.failed_paths),
            }

        # Task 2.3 — Qdrant retrieval + prompt assembly
        if fetch_result.fallback:
            logger.info("large_pr_macro_path repo=%s pr=%s", repo_id, pr_number)
            _emit_stage_log(trace_id, "embedding", "skipped", 0.0, {"reason": "large_pr_fallback"})
            _emit_stage_log(trace_id, "qdrant_retrieval", "skipped", 0.0, {"reason": "large_pr_fallback"})
            prompt = build_macro_prompt(payload)
        else:
            def _on_stage(stage: str, latency_ms: float, metadata: dict) -> None:
                _emit_stage_log(trace_id, stage, "success", latency_ms, metadata)

            def _retrieve_in_thread():
                return retrieve_and_assemble(
                    fetch_result.files, payload, _get_qdrant(), _get_embedder(),
                    on_stage=_on_stage,
                )
            pr_obj = await asyncio.get_event_loop().run_in_executor(None, _retrieve_in_thread)
            prompt = pr_obj.prompt

        # Task 2.4 — inference handoff
        logger.info("inference_start repo=%s pr=%s", repo_id, pr_number)
        with _stage_timer(trace_id, "inference_handoff") as ctx:
            review = await call_inference(prompt, _PHI_ENDPOINT, _PHI_KEY, phi_model=_PHI_DEPLOYMENT)
            ctx["metadata"] = {"review_chars": len(review)}
        inference_latency_ms = ctx["latency_ms"]
        logger.info("inference_complete repo=%s pr=%s chars=%d", repo_id, pr_number, len(review))

        # Task 2.4 — GitHub callback
        await post_github_comment(repo_id, pr_number, review, _PAT)
        logger.info("review_complete repo=%s pr=%s trace_id=%s", repo_id, pr_number, trace_id)

    except Exception as exc:
        status = "failed"
        logger.error(
            "review_failed repo=%s pr=%s trace_id=%s error=%s",
            repo_id, pr_number, trace_id, exc, exc_info=True,
        )
        try:
            await post_github_comment(repo_id, pr_number, FALLBACK_COMMENT, _PAT)
        except Exception as cb_exc:
            logger.error("fallback_comment_failed repo=%s pr=%s error=%s", repo_id, pr_number, cb_exc)

    finally:
        end_to_end_latency_ms = (time.monotonic() - t_pipeline_start) * 1000
        _emit_system_metrics(trace_id, status, end_to_end_latency_ms, inference_latency_ms)
        _review_in_progress.discard(repo_id)  # always release the concurrency guard


async def _process_sync(payload: dict[str, Any]) -> None:
    """
    Delete-then-upsert sync pipeline — runs in the background after 202 (Task 4.2).

    Every changed file (modified, added, renamed, or removed) has its stale
    vectors blind-deleted first, scoped to repo_id + file_path. Renamed files
    additionally delete under their previous_path so the old path's vectors
    don't orphan. Modified/added/renamed files are then re-fetched,
    re-chunked, re-embedded, and upserted stamped with the new commit_hash
    under the new path. Removed files stop after the delete step. Because
    delete always runs before upsert for a given path, redelivering the same
    merge event is idempotent — it never accumulates duplicate vectors.
    """
    import asyncio

    repo_id       = payload.get("repo_id", "")
    commit_hash   = payload.get("commit_hash", "")
    changed_files = payload.get("changed_files") or []

    logger.info(
        "sync_started repo=%s commit=%s files=%d",
        repo_id, commit_hash, len(changed_files),
    )

    if not repo_id:
        logger.error("sync_aborted reason=missing_repo_id")
        return

    if not changed_files:
        logger.info("sync_noop repo=%s reason=no_changed_files", repo_id)
        return

    valid_entries = [f for f in changed_files if isinstance(f, dict) and f.get("path")]
    if len(valid_entries) < len(changed_files):
        logger.warning(
            "sync_malformed_entries_skipped repo=%s count=%d",
            repo_id, len(changed_files) - len(valid_entries),
        )

    client = _get_qdrant()

    # Blind delete — every changed/deleted file loses its stale vectors before re-ingestion.
    # Renames additionally delete under previous_path so the old path never orphans.
    def _delete_all():
        for entry in valid_entries:
            _delete_file_vectors(client, repo_id, entry["path"])
            if entry.get("status") == "renamed":
                previous_path = entry.get("previous_path")
                if previous_path:
                    _delete_file_vectors(client, repo_id, previous_path)
                else:
                    logger.warning(
                        "sync_rename_missing_previous_path repo=%s path=%s",
                        repo_id, entry["path"],
                    )

    await asyncio.get_event_loop().run_in_executor(None, _delete_all)

    fetch_result = await extract_and_fetch({**payload, "changed_files": valid_entries}, pat=_PAT)

    if fetch_result.fallback:
        logger.warning("sync_fetch_skipped repo=%s reason=too_many_files", repo_id)
        return

    if not fetch_result.files:
        logger.info(
            "sync_complete repo=%s commit=%s deleted=%d upserted=0",
            repo_id, commit_hash, len(valid_entries),
        )
        return

    def _chunk_embed_upsert() -> int:
        embedder = _get_embedder()
        all_chunks = []
        for path, content in fetch_result.files.items():
            all_chunks.extend(_chunk_file(path, content))
        if not all_chunks:
            return 0
        vectors = _embed_chunks(all_chunks, embedder)
        _upsert_batch(client, all_chunks, vectors, repo_id, commit_hash)
        return len(all_chunks)

    chunks_upserted = await asyncio.get_event_loop().run_in_executor(None, _chunk_embed_upsert)

    logger.info(
        "sync_complete repo=%s commit=%s deleted=%d upserted=%d",
        repo_id, commit_hash, len(valid_entries), chunks_upserted,
    )


ZIPBALL_TIMEOUT_S: float = 120.0  # full-repo archive; slower than a single blob fetch


async def _download_zipball(repo_id: str, branch: str, pat: str) -> bytes:
    """Fetch a repository snapshot as a ZIP archive via the GitHub API (no git clone)."""
    url = f"https://api.github.com/repos/{repo_id}/zipball/{branch}"
    async with httpx.AsyncClient(timeout=ZIPBALL_TIMEOUT_S, follow_redirects=True) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {pat}"})
        resp.raise_for_status()
        return resp.content


def _extract_zipball(zip_bytes: bytes, dest_dir: str) -> str:
    """
    Extract a GitHub zipball into dest_dir and return the repo root.

    GitHub zipballs wrap all content in a single top-level directory named
    like "owner-repo-<sha>/" — unwrap it so the returned path is the actual
    repo root that ingest_repository expects.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)
    entries = [os.path.join(dest_dir, name) for name in os.listdir(dest_dir)]
    top_level_dirs = [e for e in entries if os.path.isdir(e)]
    if len(top_level_dirs) == 1:
        return top_level_dirs[0]
    return dest_dir


async def _process_reindex(payload: dict[str, Any]) -> None:
    """
    Weekly full-repository rebuild — runs in the background after 202 (Task 4.3).

    Acts as a reconciliation mechanism for missed webhooks, failed ingestion
    jobs, and metadata drift: issues a global DELETE for every vector under
    repo_id (not scoped to individual files, so files deleted since the last
    successful sync are cleaned up too), then re-fetches the repo as a ZIP
    archive from the GitHub API (no git clone — avoids disk/binary
    dependencies), re-ingests it from a temp directory, and aggressively
    wipes that temp directory afterward regardless of outcome.
    """
    import asyncio

    repo_id = payload.get("repo_id", "")
    branch  = payload.get("branch", "main")

    if not repo_id:
        logger.error("reindex_aborted reason=missing_repo_id")
        return

    logger.info("reindex_started repo=%s branch=%s", repo_id, branch)

    client = _get_qdrant()

    def _global_delete() -> None:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="repo_id",
                            match=qdrant_models.MatchValue(value=repo_id),
                        ),
                    ]
                )
            ),
        )

    tmp_dir: str | None = None
    try:
        await asyncio.get_event_loop().run_in_executor(None, _global_delete)
        from pathlib import Path as _Path

        loop = asyncio.get_event_loop()
        commit_hash   = payload.get("commit_hash", "weekly-rebuild")
        embedder      = _get_embedder()

        zip_bytes = await _download_zipball(repo_id, branch, _PAT)
        tmp_dir = tempfile.mkdtemp(prefix="pattern_buddy_reindex_")
        repo_root = _extract_zipball(zip_bytes, tmp_dir)

        # Walk + chunk (fast, I/O-light) in one executor hop.
        def _walk_and_chunk():
            files = _walk_repo(_Path(repo_root))
            chunks = []
            for rel_path, source_text in files:
                for c in _chunk_file(rel_path, source_text):
                    chunks.append(c)
            return len(files), chunks

        files_count, all_chunks = await loop.run_in_executor(None, _walk_and_chunk)
        logger.info("reindex_chunked repo=%s files=%d chunks=%d", repo_id, files_count, len(all_chunks))

        # Embed + upsert in small batches, awaiting each separately. Returning to
        # the event loop between batches lets uvicorn refresh its gunicorn liveness
        # heartbeat; a single long executor call starves it and the arbiter
        # SIGKILLs the worker at ~30s (root cause of the reindex "OOM" — memory
        # was never the issue; measured peak <1GB).
        REINDEX_BATCH = 16
        chunks_upserted = 0

        def _embed_upsert(batch):
            vectors = _embed_chunks(batch, embedder)
            _upsert_batch(client, batch, vectors, repo_id, commit_hash)
            return len(batch)

        for i in range(0, len(all_chunks), REINDEX_BATCH):
            batch = all_chunks[i:i + REINDEX_BATCH]
            chunks_upserted += await loop.run_in_executor(None, _embed_upsert, batch)
            await asyncio.sleep(0)  # yield so the heartbeat task can run

        logger.info(
            "reindex_complete repo=%s files=%d chunks=%d",
            repo_id, files_count, chunks_upserted,
        )
    except Exception as exc:
        logger.error("reindex_failed repo=%s error=%s", repo_id, exc, exc_info=True)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _reindex_in_progress.discard(repo_id)  # always release the concurrency guard


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/review", status_code=202)
async def post_review(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
) -> JSONResponse:
    """
    Receive a PR trigger from the GitHub Actions workflow.

    Security gate: validates HMAC-SHA256 signature before any processing.
    Returns 202 immediately; all orchestration runs in the background.

    Task 5.1: the trace_id is generated here, at webhook ingress, so it
    covers the full pipeline from the moment the request is accepted.
    """
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    payload: dict[str, Any] = await request.json()
    repo_id = payload.get("repo_id", "")

    # Concurrency guard: serialize reviews per repo so overlapping triggers don't
    # cannibalise the rate-limited inference endpoint. Check-and-add is atomic
    # (no await between); the background task releases it in its finally.
    if repo_id and repo_id in _review_in_progress:
        logger.info("review_rejected_already_running repo=%s pr=%s",
                    repo_id, payload.get("pr_number"))
        return JSONResponse(status_code=409, content={"status": "already_running"})
    if repo_id:
        _review_in_progress.add(repo_id)

    payload["trace_id"] = _new_trace_id(repo_id, payload.get("pr_number", 0))
    background_tasks.add_task(_process_review, payload)

    return JSONResponse(status_code=202, content={"status": "accepted"})


@app.post("/sync", status_code=202)
async def post_sync(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
) -> JSONResponse:
    """
    Receive a merge trigger from the GitHub Actions workflow (Task 4.1).

    Security gate: validates HMAC-SHA256 signature before any processing.
    Only merged PRs (action == "closed" and merged == true) proceed to the
    background sync task; closed-but-unmerged (abandoned) PRs are
    acknowledged and dropped without touching Qdrant.
    """
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    payload: dict[str, Any] = await request.json()

    if payload.get("action") != "closed" or not payload.get("merged"):
        logger.info(
            "sync_skipped repo=%s action=%s merged=%s",
            payload.get("repo_id"), payload.get("action"), payload.get("merged"),
        )
        return JSONResponse(status_code=202, content={"status": "skipped"})

    background_tasks.add_task(_process_sync, payload)
    return JSONResponse(status_code=202, content={"status": "accepted"})


@app.post("/admin/reindex", status_code=202)
async def post_admin_reindex(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
) -> JSONResponse:
    """
    Trigger a full repository rebuild (Task 4.3).

    Driven by a weekly GitHub Actions cron workflow (Sundays 03:00 UTC), and
    also serves as the manual re-trigger path for recovering from webhook
    delivery failures. Security gate: validates HMAC-SHA256 signature before
    any processing. Returns 202 immediately; the rebuild runs in the
    background since a full repo re-ingestion can take well over GitHub's
    CI timeout.
    """
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    payload: dict[str, Any] = await request.json()
    repo_id = payload.get("repo_id", "")

    # Concurrency guard: reject a second reindex for a repo already being rebuilt.
    if repo_id and repo_id in _reindex_in_progress:
        logger.info("reindex_rejected_already_running repo=%s", repo_id)
        return JSONResponse(status_code=409, content={"status": "already_running"})
    if repo_id:
        _reindex_in_progress.add(repo_id)

    background_tasks.add_task(_process_reindex, payload)

    return JSONResponse(status_code=202, content={"status": "accepted"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

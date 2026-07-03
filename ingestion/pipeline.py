"""
Phase 1 — Ingestion pipeline: file router + embedding + Qdrant upsert.

Entry point:
    ingest_repository(repo_path, repo_id, commit_hash, qdrant_client)

Steps:
1. Walk the repository directory tree.
2. Classify each file into Track A (.py), Track B (.md/.txt), Track C (.toml/.json/.yaml).
3. Skip binaries, lockfiles, generated files, and the .git/ directory.
4. Chunk each file with the appropriate chunker.
5. Embed all chunks in batch using nomic-embed-text (fastembed / ONNX).
6. Stamp each chunk with repo_id and commit_hash, then upsert to Qdrant.

Multitenancy: every point carries a repo_id payload, enforced at the collection
level with an is_tenant index. Cross-tenant retrieval is impossible via the
middleware filter defined in Phase 2.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Union

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from ingestion.python_chunker import chunk_python_file, MAX_TOKENS
from ingestion.prose_chunker import chunk_prose_file
from ingestion.config_chunker import chunk_config_file

logger = logging.getLogger(__name__)

# ── Routing tables ────────────────────────────────────────────────────────────

TRACK_A_EXTS = {".py"}
TRACK_B_EXTS = {".md", ".txt"}
TRACK_C_EXTS = {".toml", ".json", ".yaml", ".yml"}

SKIP_FILENAMES = {
    "poetry.lock", "package-lock.json", "yarn.lock", "uv.lock",
    "Pipfile.lock", "composer.lock", "Gemfile.lock", "cargo.lock",
    "packages.lock.json",
}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".tox", "dist", "build",
}

QDRANT_COLLECTION = "global_codebase_memory"


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except OSError:
        return True


def _should_skip(path: Path) -> bool:
    if path.name.lower() in {n.lower() for n in SKIP_FILENAMES}:
        return True
    if path.suffix.lower() == ".lock":
        return True
    return False


def _classify(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in TRACK_A_EXTS:
        return "A"
    if ext in TRACK_B_EXTS:
        return "B"
    if ext in TRACK_C_EXTS:
        return "C"
    return None


def _walk_repo(repo_path: Path) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for file_path in sorted(repo_path.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        if _should_skip(file_path):
            continue
        if _classify(file_path) is None:
            continue
        if _is_binary(file_path):
            continue
        try:
            source_text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            continue
        rel_path = str(file_path.relative_to(repo_path))
        results.append((rel_path, source_text))
    return results


def _chunk_file(rel_path: str, source_text: str):
    track = _classify(Path(rel_path))
    if track == "A":
        return chunk_python_file(rel_path, source_text)
    if track == "B":
        return chunk_prose_file(rel_path, source_text)
    if track == "C":
        return chunk_config_file(rel_path, source_text)
    return []


def _load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")


def _embed_chunks(chunks, embedder) -> list[list[float]]:
    texts = [c.content for c in chunks]
    return [v.tolist() for v in embedder.embed(texts)]


def _delete_file_vectors(client, repo_id: str, file_path: str) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="repo_id",
                        match=qdrant_models.MatchValue(value=repo_id),
                    ),
                    qdrant_models.FieldCondition(
                        key="file_path",
                        match=qdrant_models.MatchValue(value=file_path),
                    ),
                ]
            )
        ),
    )


def _upsert_batch(client, chunks, vectors, repo_id, commit_hash):
    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "repo_id":        repo_id,
            "file_path":      chunk.file_path,
            "file_type":      chunk.file_type,
            "chunk_strategy": chunk.chunk_strategy,
            "target_module":  getattr(chunk, "target_module", None),
            "commit_hash":    commit_hash,
            "content":        chunk.content,
        }
        points.append(qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def ingest_repository(repo_path, repo_id, commit_hash, qdrant_client, batch_size=64, embedder=None):
    """
    embedder: pass an already-loaded TextEmbedding instance to reuse a shared
    singleton (e.g. the API's cached embedder) instead of loading a second,
    independent ~1GB copy of the model. Only loads its own when omitted,
    which is what the standalone run_ingestion.py CLI relies on.
    """
    repo_path = Path(repo_path)
    if embedder is None:
        embedder = _load_embedder()
    files = _walk_repo(repo_path)
    logger.info("Found %d ingestible files in %s", len(files), repo_path)

    all_chunks = []
    files_skipped = 0
    for rel_path, source_text in files:
        chunks = _chunk_file(rel_path, source_text)
        if not chunks:
            files_skipped += 1
            continue
        _delete_file_vectors(qdrant_client, repo_id, rel_path)
        for c in chunks:
            c.commit_hash = commit_hash
        all_chunks.extend(chunks)

    chunks_upserted = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        vectors = _embed_chunks(batch, embedder)
        _upsert_batch(qdrant_client, batch, vectors, repo_id, commit_hash)
        chunks_upserted += len(batch)

    return {
        "files_processed": len(files) - files_skipped,
        "chunks_upserted": chunks_upserted,
        "files_skipped":   files_skipped,
    }

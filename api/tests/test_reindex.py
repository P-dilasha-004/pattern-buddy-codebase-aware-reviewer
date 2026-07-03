"""
Tests for Task 4.3 — POST /admin/reindex weekly rebuild endpoint and
_process_reindex / _extract_zipball helpers.

Covers:
- Endpoint: HMAC signature validation, 202 response, background task spawned
- _process_reindex: global (repo-wide, not per-file) delete before re-ingestion
- _process_reindex: missing repo_id aborts before any Qdrant/GitHub call
- _process_reindex: temp directory is cleaned up after success
- _process_reindex: temp directory is cleaned up even when ingestion raises
- _process_reindex: zipball download failure is caught, doesn't crash the task
- _extract_zipball: unwraps GitHub's single top-level "owner-repo-<sha>/" dir
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SECRET = "test-webhook-secret-abc123"


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    import importlib
    import api.main as m
    importlib.reload(m)
    monkeypatch.setattr("api.main._WEBHOOK_SECRET", SECRET)


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app, raise_server_exceptions=True)


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


PAYLOAD = {"repo_id": "owner/repo", "branch": "main"}


# ── Endpoint ──────────────────────────────────────────────────────────────────

class TestReindexEndpoint:
    def test_returns_202(self, client):
        body = json.dumps(PAYLOAD).encode()
        resp = client.post(
            "/admin/reindex",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202

    def test_background_task_spawned_with_payload(self, client):
        body = json.dumps(PAYLOAD).encode()
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = client.post(
                "/admin/reindex",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        assert resp.status_code == 202
        mock_task.assert_called_once()
        assert mock_task.call_args[0][0]["repo_id"] == "owner/repo"

    def test_missing_signature_returns_401(self, client):
        body = json.dumps(PAYLOAD).encode()
        resp = client.post(
            "/admin/reindex",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self, client):
        body = json.dumps(PAYLOAD).encode()
        resp = client.post(
            "/admin/reindex",
            content=body,
            headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_unauthorized_request_does_not_spawn_task(self, client):
        body = json.dumps(PAYLOAD).encode()
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            client.post(
                "/admin/reindex",
                content=body,
                headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()


# ── _extract_zipball ──────────────────────────────────────────────────────────

def _build_zip_with_top_level_dir(dirname: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel_path, content in files.items():
            zf.writestr(f"{dirname}/{rel_path}", content)
    return buf.getvalue()


class TestExtractZipball:
    def test_unwraps_single_top_level_directory(self, tmp_path):
        from api.main import _extract_zipball
        zip_bytes = _build_zip_with_top_level_dir(
            "owner-repo-abc123", {"README.md": "# hi", "src/main.py": "pass"}
        )
        repo_root = _extract_zipball(zip_bytes, str(tmp_path))

        assert os.path.basename(repo_root) == "owner-repo-abc123"
        assert os.path.isfile(os.path.join(repo_root, "README.md"))
        assert os.path.isfile(os.path.join(repo_root, "src", "main.py"))

    def test_returns_dest_dir_when_no_single_top_level_dir(self, tmp_path):
        from api.main import _extract_zipball
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("loose_file.txt", "content")  # no wrapping directory
        repo_root = _extract_zipball(buf.getvalue(), str(tmp_path))

        assert repo_root == str(tmp_path)
        assert os.path.isfile(os.path.join(repo_root, "loose_file.txt"))


# ── _process_reindex ──────────────────────────────────────────────────────────

class TestProcessReindex:
    @pytest.mark.asyncio
    async def test_missing_repo_id_aborts_before_any_call(self):
        import api.main as m
        with patch("api.main._get_qdrant") as mock_qdrant, \
             patch("api.main._download_zipball", new_callable=AsyncMock) as mock_dl:
            await m._process_reindex({"branch": "main"})
        mock_qdrant.assert_not_called()
        mock_dl.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_is_repo_wide_not_scoped_to_a_file(self):
        """
        Task 4.3 explicitly calls for a *global* DELETE by repo_id (unlike
        4.2's per-file delete) so files removed since the last successful
        sync don't orphan. Assert the filter has no file_path condition.
        """
        import api.main as m
        fake_client = MagicMock()

        with patch("api.main._get_qdrant", return_value=fake_client), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes"), \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo"), \
             patch("api.main.ingest_repository",
                   return_value={"files_processed": 3, "chunks_upserted": 10, "files_skipped": 0}), \
             patch("shutil.rmtree") as mock_rmtree:
            await m._process_reindex({"repo_id": "owner/repo", "branch": "main"})

        fake_client.delete.assert_called_once()
        _, kwargs = fake_client.delete.call_args
        conditions = kwargs["points_selector"].filter.must
        keys = {c.key for c in conditions}
        assert keys == {"repo_id"}  # no file_path condition — this is a full tenant wipe
        assert conditions[0].match.value == "owner/repo"
        mock_rmtree.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_repository_called_with_extracted_path_and_repo_id(self):
        import api.main as m
        fake_client = MagicMock()
        fake_embedder = MagicMock()

        with patch("api.main._get_qdrant", return_value=fake_client), \
             patch("api.main._get_embedder", return_value=fake_embedder), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes"), \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo_root") as mock_extract, \
             patch("api.main.ingest_repository",
                   return_value={"files_processed": 1, "chunks_upserted": 2, "files_skipped": 0}) as mock_ingest, \
             patch("shutil.rmtree"):
            await m._process_reindex({"repo_id": "owner/repo", "branch": "develop",
                                       "commit_hash": "deadbeef"})

        mock_extract.assert_called_once_with(b"zipbytes", mock_extract.call_args[0][1])
        mock_ingest.assert_called_once()
        _, kwargs = mock_ingest.call_args
        assert kwargs["repo_path"] == "/tmp/fake_repo_root"
        assert kwargs["repo_id"] == "owner/repo"
        assert kwargs["commit_hash"] == "deadbeef"
        assert kwargs["qdrant_client"] is fake_client

    @pytest.mark.asyncio
    async def test_ingest_repository_reuses_shared_embedder_not_a_fresh_one(self):
        """
        Regression test: ingest_repository must be called with the API's
        cached singleton embedder, not left to load its own independent
        copy of the ~1GB model. Loading a second copy on top of whatever
        /review or /sync already cached is what caused the OOM kills.
        """
        import api.main as m
        fake_embedder = MagicMock()

        with patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=fake_embedder) as mock_get_embedder, \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes"), \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo"), \
             patch("api.main.ingest_repository",
                   return_value={"files_processed": 1, "chunks_upserted": 1, "files_skipped": 0}) as mock_ingest, \
             patch("shutil.rmtree"):
            await m._process_reindex({"repo_id": "owner/repo"})

        mock_get_embedder.assert_called_once()
        _, kwargs = mock_ingest.call_args
        assert kwargs["embedder"] is fake_embedder

    @pytest.mark.asyncio
    async def test_temp_dir_cleaned_up_after_success(self):
        import api.main as m
        with patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes"), \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo"), \
             patch("api.main.ingest_repository",
                   return_value={"files_processed": 1, "chunks_upserted": 1, "files_skipped": 0}), \
             patch("shutil.rmtree") as mock_rmtree, \
             patch("tempfile.mkdtemp", return_value="/tmp/pattern_buddy_reindex_xyz"):
            await m._process_reindex({"repo_id": "owner/repo"})

        mock_rmtree.assert_called_once_with("/tmp/pattern_buddy_reindex_xyz", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_temp_dir_cleaned_up_even_when_ingestion_raises(self):
        import api.main as m
        with patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes"), \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo"), \
             patch("api.main.ingest_repository", side_effect=RuntimeError("ingest boom")), \
             patch("shutil.rmtree") as mock_rmtree, \
             patch("tempfile.mkdtemp", return_value="/tmp/pattern_buddy_reindex_xyz"):
            await m._process_reindex({"repo_id": "owner/repo"})  # must not raise

        mock_rmtree.assert_called_once_with("/tmp/pattern_buddy_reindex_xyz", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_zipball_download_failure_does_not_crash_task(self):
        import api.main as m
        with patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   side_effect=RuntimeError("network boom")) as mock_dl, \
             patch("api.main.ingest_repository") as mock_ingest, \
             patch("shutil.rmtree") as mock_rmtree:
            await m._process_reindex({"repo_id": "owner/repo"})  # must not raise

        mock_dl.assert_called_once()
        mock_ingest.assert_not_called()
        mock_rmtree.assert_not_called()  # tmp_dir was never created

    @pytest.mark.asyncio
    async def test_defaults_to_main_branch_when_omitted(self):
        import api.main as m
        with patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main._download_zipball", new_callable=AsyncMock,
                   return_value=b"zipbytes") as mock_dl, \
             patch("api.main._extract_zipball", return_value="/tmp/fake_repo"), \
             patch("api.main.ingest_repository",
                   return_value={"files_processed": 0, "chunks_upserted": 0, "files_skipped": 0}), \
             patch("shutil.rmtree"):
            await m._process_reindex({"repo_id": "owner/repo"})  # no "branch" key

        mock_dl.assert_called_once_with("owner/repo", "main", m._PAT)

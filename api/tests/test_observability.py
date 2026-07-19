"""
Tests for Task 5.1 — trace ID generation & structured stage logging.

Covers:
- _new_trace_id format: pr_{repo}_{pr_number}_{timestamp} (planning.md 9.2)
- repo_id with "/" is sanitized for use in the trace_id / log filenames
- _emit_stage_log: JSON schema (planning.md 9.6), appends JSONL to logs/<stage>.json
- _stage_timer: success path logs status=success with metadata and timing
- _stage_timer: exception path logs status=failed and re-raises
- POST /review: trace_id is generated at webhook ingress and forwarded to
  the background task
- _process_review: all four Task 5.1 stages (diff_fetch, embedding,
  qdrant_retrieval, inference_handoff) get logged under the same trace_id,
  including the "skipped" embedding/qdrant_retrieval entries on the
  large-PR macro-summary fallback path (exit criterion: "every review
  produces a complete trace log")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.diff import FetchResult
from api.retrieval import PromptResult

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


REVIEW_PAYLOAD = {
    "repo_id": "owner/repo",
    "pr_number": 42,
    "base_sha": "aaa111",
    "head_sha": "bbb222",
    "changed_files": [
        {"path": "src/auth.py", "status": "modified",
         "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/auth.py"}
    ],
}


# ── _new_trace_id ─────────────────────────────────────────────────────────────

class TestNewTraceId:
    def test_format_matches_spec(self):
        from api.main import _new_trace_id
        trace_id = _new_trace_id("owner/repo", 42)
        assert trace_id.startswith("pr_owner-repo_42_")
        suffix = trace_id.rsplit("_", 1)[1]
        assert suffix.isdigit()

    def test_slash_in_repo_id_is_sanitized(self):
        from api.main import _new_trace_id
        trace_id = _new_trace_id("owner/repo", 1)
        assert "/" not in trace_id

    def test_timestamp_is_recent(self):
        from api.main import _new_trace_id
        before = int(time.time())
        trace_id = _new_trace_id("owner/repo", 1)
        after = int(time.time())
        ts = int(trace_id.rsplit("_", 1)[1])
        assert before <= ts <= after

    def test_missing_repo_id_falls_back_to_unknown(self):
        from api.main import _new_trace_id
        trace_id = _new_trace_id("", 1)
        assert trace_id.startswith("pr_unknown_1_")


# ── _emit_stage_log ───────────────────────────────────────────────────────────

class TestEmitStageLog:
    def test_writes_jsonl_entry_to_stage_file(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_stage_log("trace-1", "embedding", "success", 142.456, {"chunks": 12})

        log_file = tmp_path / "embedding.json"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry == {
            "trace_id": "trace-1",
            "stage": "embedding",
            "status": "success",
            "latency_ms": 142.46,
            "metadata": {"chunks": 12},
        }

    def test_appends_multiple_entries_as_separate_lines(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_stage_log("trace-1", "diff_fetch", "success", 10.0, {})
        m._emit_stage_log("trace-2", "diff_fetch", "success", 20.0, {})

        lines = (tmp_path / "diff_fetch.json").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["trace_id"] == "trace-1"
        assert json.loads(lines[1])["trace_id"] == "trace-2"

    def test_separate_stages_go_to_separate_files(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_stage_log("trace-1", "embedding", "success", 1.0, {})
        m._emit_stage_log("trace-1", "qdrant_retrieval", "success", 2.0, {})

        assert (tmp_path / "embedding.json").exists()
        assert (tmp_path / "qdrant_retrieval.json").exists()

    def test_none_metadata_defaults_to_empty_dict(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        m._emit_stage_log("trace-1", "inference_handoff", "success", 5.0, None)

        entry = json.loads((tmp_path / "inference_handoff.json").read_text().strip())
        assert entry["metadata"] == {}

    def test_disk_write_failure_does_not_raise(self, monkeypatch):
        import api.main as m
        # Point LOGS_DIR somewhere mkdir can't create (root-owned path segment).
        monkeypatch.setattr(m, "LOGS_DIR", __import__("pathlib").Path("/nonexistent_root_only/logs"))
        m._emit_stage_log("trace-1", "diff_fetch", "success", 1.0, {})  # must not raise


# ── _stage_timer ──────────────────────────────────────────────────────────────

class TestStageTimer:
    def test_success_path_logs_status_success_with_metadata(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with m._stage_timer("trace-1", "diff_fetch") as ctx:
            ctx["metadata"] = {"files_fetched": 3}

        entry = json.loads((tmp_path / "diff_fetch.json").read_text().strip())
        assert entry["status"] == "success"
        assert entry["metadata"] == {"files_fetched": 3}
        assert entry["latency_ms"] >= 0

    def test_exception_path_logs_status_failed_and_reraises(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with pytest.raises(ValueError):
            with m._stage_timer("trace-1", "inference_handoff") as ctx:
                ctx["metadata"] = {"attempted": True}
                raise ValueError("boom")

        entry = json.loads((tmp_path / "inference_handoff.json").read_text().strip())
        assert entry["status"] == "failed"
        assert entry["metadata"] == {"attempted": True}


# ── POST /review generates trace_id at ingress ───────────────────────────────

class TestTraceIdAtIngress:
    def test_trace_id_forwarded_to_background_task(self, client):
        body = json.dumps(REVIEW_PAYLOAD).encode()
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            client.post(
                "/review",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        called_with = mock_task.call_args[0][0]
        assert "trace_id" in called_with
        assert called_with["trace_id"].startswith("pr_owner-repo_42_")


# ── _process_review emits a complete trace across all 4 stages ──────────────

class TestProcessReviewTraceCompleteness:
    @pytest.mark.asyncio
    async def test_normal_path_logs_all_four_stages(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        fetch_result = FetchResult(files={"src/auth.py": "def foo(): pass"})
        prompt_result = PromptResult(prompt="PROMPT", chunks_used=1, chunks_retrieved=1)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock, return_value=fetch_result), \
             patch("api.main.retrieve_and_assemble") as mock_retrieve, \
             patch("api.main.call_inference", new_callable=AsyncMock, return_value="REVIEW TEXT"), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):

            def _fake_retrieve(files, payload, client_, embedder, on_stage=None):
                if on_stage:
                    on_stage("embedding", 5.0, {"file_count": 1})
                    on_stage("qdrant_retrieval", 8.0, {"chunks_retrieved": 1})
                return prompt_result
            mock_retrieve.side_effect = _fake_retrieve

            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-xyz"})

        for stage in ("diff_fetch", "embedding", "qdrant_retrieval", "inference_handoff"):
            log_file = tmp_path / f"{stage}.json"
            assert log_file.exists(), f"missing trace log for stage {stage}"
            entry = json.loads(log_file.read_text().strip().splitlines()[-1])
            assert entry["trace_id"] == "trace-xyz"
            assert entry["status"] == "success"

    @pytest.mark.asyncio
    async def test_macro_fallback_path_still_logs_all_four_stages(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        fetch_result = FetchResult(fallback=True)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock, return_value=fetch_result), \
             patch("api.main.build_macro_prompt", return_value="MACRO PROMPT"), \
             patch("api.main.call_inference", new_callable=AsyncMock, return_value="REVIEW TEXT"), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):
            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-fallback"})

        for stage in ("diff_fetch", "embedding", "qdrant_retrieval", "inference_handoff"):
            log_file = tmp_path / f"{stage}.json"
            assert log_file.exists(), f"missing trace log for stage {stage}"

        embedding_entry = json.loads((tmp_path / "embedding.json").read_text().strip())
        assert embedding_entry["status"] == "skipped"
        qdrant_entry = json.loads((tmp_path / "qdrant_retrieval.json").read_text().strip())
        assert qdrant_entry["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_failure_mid_pipeline_still_logs_failed_stage(self, tmp_path, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "LOGS_DIR", tmp_path)

        with patch("api.main.extract_and_fetch", new_callable=AsyncMock,
                   side_effect=RuntimeError("fetch boom")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):
            await m._process_review({**REVIEW_PAYLOAD, "trace_id": "trace-fail"})

        entry = json.loads((tmp_path / "diff_fetch.json").read_text().strip())
        assert entry["status"] == "failed"
        assert entry["trace_id"] == "trace-fail"

"""
Tests for the unified POST /webhook endpoint (GitHub App conversion).

GitHub Apps deliver every event type to one URL, distinguished by the
X-GitHub-Event header rather than payload shape or a separate endpoint per
trigger — this replaces the old /review (which depended on a GitHub Actions
workflow in the target repo building a custom JSON payload).

Covers:
- Valid HMAC signature → 202 Accepted
- Missing / wrong / malformed signature → 401
- ping event → 200
- pull_request opened/synchronize → fetches PR files via the installation
  token, spawns _process_review with the internal payload shape
- pull_request closed+merged → spawns _process_sync
- pull_request closed (not merged) / other actions → 202 "skipped", no task
- Missing installation/repository context → 202 "ignored", no task
- Review concurrency guard (409) on pull_request opened/synchronize
- GET /health returns 200
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

SECRET = "test-webhook-secret-abc123"
INSTALLATION_ID = 999999


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    # Re-import after env is set so the module-level _WEBHOOK_SECRET picks it up
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


def _pr_payload(action="opened", pr_number=42, merged=False, repo="owner/repo"):
    return {
        "action": action,
        "number": pr_number,
        "pull_request": {
            "base": {"sha": "aaa111"},
            "head": {"sha": "bbb222"},
            "title": "Add auth service",
            "merged": merged,
            "merge_commit_sha": "ccc333",
        },
        "repository": {"full_name": repo},
        "installation": {"id": INSTALLATION_ID},
    }


def _post_webhook(client, payload, event, secret=SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "x-hub-signature-256": _sign(body, secret),
            "x-github-event": event,
            "content-type": "application/json",
        },
    )


# ── ping ──────────────────────────────────────────────────────────────────────

class TestPing:
    def test_ping_returns_200(self, client):
        resp = _post_webhook(client, {"zen": "hi"}, "ping")
        assert resp.status_code == 200
        assert resp.json() == {"status": "pong"}


# ── pull_request: opened/synchronize -> review ───────────────────────────────

class TestPullRequestReview:
    def test_opened_returns_202_and_spawns_review(self, client):
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task, \
             patch("api.main.get_installation_token", new_callable=AsyncMock, return_value="tok"), \
             patch("api.main.fetch_pr_files", new_callable=AsyncMock,
                   return_value=[{"path": "src/a.py", "status": "modified"}]) as mock_files:
            resp = _post_webhook(client, _pr_payload("opened"), "pull_request")

        assert resp.status_code == 202
        assert resp.json() == {"status": "accepted"}
        mock_task.assert_called_once()
        mock_files.assert_called_once_with("owner/repo", 42, "tok")

        called_with = mock_task.call_args[0][0]
        assert called_with["repo_id"] == "owner/repo"
        assert called_with["pr_number"] == 42
        assert called_with["head_sha"] == "bbb222"
        assert called_with["base_sha"] == "aaa111"
        assert called_with["commit_message"] == "Add auth service"
        assert called_with["installation_id"] == INSTALLATION_ID
        assert called_with["changed_files"] == [{"path": "src/a.py", "status": "modified"}]
        assert "trace_id" in called_with

    def test_synchronize_also_triggers_review(self, client):
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task, \
             patch("api.main.get_installation_token", new_callable=AsyncMock, return_value="tok"), \
             patch("api.main.fetch_pr_files", new_callable=AsyncMock, return_value=[]):
            resp = _post_webhook(client, _pr_payload("synchronize"), "pull_request")
        assert resp.status_code == 202
        mock_task.assert_called_once()

    def test_second_review_same_repo_rejected_409(self, client):
        import api.main as m
        m._review_in_progress.add("owner/repo")
        try:
            with patch("api.main._process_review", new_callable=AsyncMock) as mock_task, \
                 patch("api.main.get_installation_token", new_callable=AsyncMock, return_value="tok"), \
                 patch("api.main.fetch_pr_files", new_callable=AsyncMock, return_value=[]):
                resp = _post_webhook(client, _pr_payload("opened"), "pull_request")
            assert resp.status_code == 409
            assert resp.json() == {"status": "already_running"}
            mock_task.assert_not_called()
        finally:
            m._review_in_progress.discard("owner/repo")

    def test_different_repo_not_blocked(self, client):
        import api.main as m
        m._review_in_progress.add("someone-else/repo")
        try:
            with patch("api.main._process_review", new_callable=AsyncMock) as mock_task, \
                 patch("api.main.get_installation_token", new_callable=AsyncMock, return_value="tok"), \
                 patch("api.main.fetch_pr_files", new_callable=AsyncMock, return_value=[]):
                resp = _post_webhook(client, _pr_payload("opened", repo="owner/repo"), "pull_request")
            assert resp.status_code == 202
            mock_task.assert_called_once()
        finally:
            m._review_in_progress.discard("someone-else/repo")
            m._review_in_progress.discard("owner/repo")


# ── pull_request: closed+merged -> sync ──────────────────────────────────────

class TestPullRequestMergeSync:
    def test_closed_and_merged_spawns_sync(self, client):
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task, \
             patch("api.main.get_installation_token", new_callable=AsyncMock, return_value="tok"), \
             patch("api.main.fetch_pr_files", new_callable=AsyncMock,
                   return_value=[{"path": "src/a.py", "status": "modified"}]):
            resp = _post_webhook(client, _pr_payload("closed", merged=True), "pull_request")

        assert resp.status_code == 202
        mock_task.assert_called_once()
        called_with = mock_task.call_args[0][0]
        assert called_with["repo_id"] == "owner/repo"
        assert called_with["commit_hash"] == "ccc333"
        assert called_with["changed_files"] == [{"path": "src/a.py", "status": "modified"}]
        assert called_with["installation_id"] == INSTALLATION_ID

    def test_closed_unmerged_is_skipped(self, client):
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_sync, \
             patch("api.main._process_review", new_callable=AsyncMock) as mock_review:
            resp = _post_webhook(client, _pr_payload("closed", merged=False), "pull_request")
        assert resp.status_code == 202
        assert resp.json() == {"status": "skipped"}
        mock_sync.assert_not_called()
        mock_review.assert_not_called()

    def test_unhandled_action_is_skipped(self, client):
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_review, \
             patch("api.main._process_sync", new_callable=AsyncMock) as mock_sync:
            resp = _post_webhook(client, _pr_payload("labeled"), "pull_request")
        assert resp.status_code == 202
        assert resp.json() == {"status": "skipped"}
        mock_review.assert_not_called()
        mock_sync.assert_not_called()


# ── Missing installation/repository context ──────────────────────────────────

class TestMissingContext:
    def test_pull_request_without_installation_is_ignored(self, client):
        payload = _pr_payload("opened")
        del payload["installation"]
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "pull_request")
        assert resp.status_code == 202
        assert resp.json() == {"status": "ignored"}
        mock_task.assert_not_called()

    def test_unhandled_event_type_is_ignored(self, client):
        resp = _post_webhook(client, {"anything": True}, "some_other_event")
        assert resp.status_code == 202
        assert resp.json() == {"status": "ignored"}


# ── Security gate — missing / invalid signature ───────────────────────────────

class TestSignatureValidation:
    def test_missing_signature_returns_401(self, client):
        body = json.dumps(_pr_payload()).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"x-github-event": "pull_request", "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self, client):
        body = json.dumps(_pr_payload()).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                     "x-github-event": "pull_request",
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_malformed_signature_no_prefix_returns_401(self, client):
        body = json.dumps(_pr_payload()).encode()
        sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"x-hub-signature-256": sig,  # missing "sha256=" prefix
                     "x-github-event": "pull_request",
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_signature_for_different_secret_returns_401(self, client):
        resp = _post_webhook(client, _pr_payload(), "pull_request", secret="wrong-secret")
        assert resp.status_code == 401

    def test_unauthorized_request_does_not_spawn_task(self, client):
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            body = json.dumps(_pr_payload()).encode()
            client.post(
                "/webhook",
                content=body,
                headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                         "x-github-event": "pull_request",
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()


# ── _process_review direct-call: concurrency guard release ──────────────────

class TestProcessReviewLockRelease:
    @pytest.mark.asyncio
    async def test_process_review_releases_lock(self):
        import api.main as m
        m._review_in_progress.add("owner/repo")
        with patch("api.main.extract_and_fetch", new_callable=AsyncMock,
                   side_effect=RuntimeError("boom")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock):
            await m._process_review({"repo_id": "owner/repo", "pr_number": 42,
                                     "trace_id": "t"})
        assert "owner/repo" not in m._review_in_progress


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_response_body(self, client):
        assert client.get("/health").json() == {"status": "ok"}

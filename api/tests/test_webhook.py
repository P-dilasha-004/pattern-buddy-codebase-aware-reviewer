"""
Tests for Task 2.1 — POST /review webhook endpoint.

Covers:
- Valid HMAC signature → 202 Accepted
- Missing signature header → 401
- Wrong signature → 401
- Malformed signature (no sha256= prefix) → 401
- Background task is spawned (not awaited synchronously)
- GET /health returns 200
- Payload is passed through to the background task
"""

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

SECRET = "test-webhook-secret-abc123"


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


SAMPLE_PAYLOAD = {
    "repo_id": "owner/repo",
    "pr_number": 42,
    "base_sha": "aaa111",
    "head_sha": "bbb222",
    "changed_files": [
        {
            "path": "src/auth.py",
            "status": "modified",
            "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/auth.py",
        }
    ],
}


# ── Happy path ────────────────────────────────────────────────────────────────

class TestValidRequest:
    def test_returns_202(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        resp = client.post(
            "/review",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202

    def test_response_body_accepted(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        resp = client.post(
            "/review",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.json() == {"status": "accepted"}

    def test_background_task_spawned(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            resp = client.post(
                "/review",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        assert resp.status_code == 202
        mock_task.assert_called_once()

    def test_payload_forwarded_to_background_task(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            client.post(
                "/review",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        called_with = mock_task.call_args[0][0]
        assert called_with["repo_id"] == "owner/repo"
        assert called_with["pr_number"] == 42
        assert called_with["head_sha"] == "bbb222"


class TestReviewConcurrencyGuard:
    def test_second_review_same_repo_rejected_409(self, client):
        import api.main as m
        m._review_in_progress.add("owner/repo")  # simulate one already running
        try:
            body = json.dumps(SAMPLE_PAYLOAD).encode()
            with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
                resp = client.post(
                    "/review",
                    content=body,
                    headers={"x-hub-signature-256": _sign(body),
                             "content-type": "application/json"},
                )
            assert resp.status_code == 409
            assert resp.json() == {"status": "already_running"}
            mock_task.assert_not_called()  # no overlapping review spawned
        finally:
            m._review_in_progress.discard("owner/repo")

    def test_different_repo_not_blocked(self, client):
        import api.main as m
        m._review_in_progress.add("someone-else/repo")
        try:
            body = json.dumps(SAMPLE_PAYLOAD).encode()  # owner/repo
            with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
                resp = client.post(
                    "/review",
                    content=body,
                    headers={"x-hub-signature-256": _sign(body),
                             "content-type": "application/json"},
                )
            assert resp.status_code == 202
            mock_task.assert_called_once()
        finally:
            m._review_in_progress.discard("someone-else/repo")
            m._review_in_progress.discard("owner/repo")

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


# ── Security gate — missing / invalid signature ───────────────────────────────

class TestSignatureValidation:
    def test_missing_signature_returns_401(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        resp = client.post(
            "/review",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        resp = client.post(
            "/review",
            content=body,
            headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_malformed_signature_no_prefix_returns_401(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        resp = client.post(
            "/review",
            content=body,
            headers={"x-hub-signature-256": sig,  # missing "sha256=" prefix
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_signature_for_different_secret_returns_401(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        wrong_sig = _sign(body, secret="wrong-secret")
        resp = client.post(
            "/review",
            content=body,
            headers={"x-hub-signature-256": wrong_sig,
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_signature_for_different_body_returns_401(self, client):
        real_body = json.dumps(SAMPLE_PAYLOAD).encode()
        tampered_body = json.dumps({**SAMPLE_PAYLOAD, "head_sha": "evil"}).encode()
        resp = client.post(
            "/review",
            content=tampered_body,
            headers={"x-hub-signature-256": _sign(real_body),  # sig of original
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_unauthorized_request_does_not_spawn_task(self, client):
        body = json.dumps(SAMPLE_PAYLOAD).encode()
        with patch("api.main._process_review", new_callable=AsyncMock) as mock_task:
            client.post(
                "/review",
                content=body,
                headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_response_body(self, client):
        assert client.get("/health").json() == {"status": "ok"}

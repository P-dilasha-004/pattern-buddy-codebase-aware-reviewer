"""
Tests for Task 4.1 — POST /sync merge webhook endpoint.

Covers:
- Valid merge event (action=closed, merged=true) → 202 Accepted, task spawned
- Closed-but-unmerged PR → 202 "skipped", no task spawned
- Non-"closed" action → 202 "skipped", no task spawned
- Missing signature header → 401
- Wrong signature → 401
- Unauthorized request never reaches the merge check or spawns a task
- GET /health still returns 200 (unaffected by this endpoint)
"""

import hashlib
import hmac
import json

import pytest
from unittest.mock import AsyncMock, patch
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


MERGED_PAYLOAD = {
    "action": "closed",
    "merged": True,
    "repo_id": "owner/repo",
    "commit_hash": "ccc333",
    "changed_files": [
        {
            "path": "src/auth.py",
            "status": "modified",
            "raw_url": "https://raw.githubusercontent.com/owner/repo/ccc333/src/auth.py",
        }
    ],
}


# ── Happy path — merged PR ───────────────────────────────────────────────────

class TestMergedRequest:
    def test_returns_202(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202

    def test_response_body_accepted(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.json() == {"status": "accepted"}

    def test_background_task_spawned(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task:
            resp = client.post(
                "/sync",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        assert resp.status_code == 202
        mock_task.assert_called_once()

    def test_payload_forwarded_to_background_task(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task:
            client.post(
                "/sync",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        called_with = mock_task.call_args[0][0]
        assert called_with["repo_id"] == "owner/repo"
        assert called_with["commit_hash"] == "ccc333"


# ── Abandoned / non-merge events are acknowledged but dropped ───────────────

class TestNonMergeEvents:
    def test_closed_but_unmerged_returns_skipped(self, client):
        payload = {**MERGED_PAYLOAD, "merged": False}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202
        assert resp.json() == {"status": "skipped"}

    def test_closed_but_unmerged_does_not_spawn_task(self, client):
        payload = {**MERGED_PAYLOAD, "merged": False}
        body = json.dumps(payload).encode()
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task:
            client.post(
                "/sync",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()

    def test_non_closed_action_returns_skipped(self, client):
        payload = {**MERGED_PAYLOAD, "action": "opened"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202
        assert resp.json() == {"status": "skipped"}

    def test_non_closed_action_does_not_spawn_task(self, client):
        payload = {**MERGED_PAYLOAD, "action": "opened"}
        body = json.dumps(payload).encode()
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task:
            client.post(
                "/sync",
                content=body,
                headers={"x-hub-signature-256": _sign(body),
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()

    def test_missing_merged_field_returns_skipped(self, client):
        payload = {k: v for k, v in MERGED_PAYLOAD.items() if k != "merged"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": _sign(body),
                     "content-type": "application/json"},
        )
        assert resp.status_code == 202
        assert resp.json() == {"status": "skipped"}


# ── Security gate — missing / invalid signature ───────────────────────────────

class TestSignatureValidation:
    def test_missing_signature_returns_401(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_malformed_signature_no_prefix_returns_401(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        resp = client.post(
            "/sync",
            content=body,
            headers={"x-hub-signature-256": sig,  # missing "sha256=" prefix
                     "content-type": "application/json"},
        )
        assert resp.status_code == 401

    def test_unauthorized_request_does_not_spawn_task(self, client):
        body = json.dumps(MERGED_PAYLOAD).encode()
        with patch("api.main._process_sync", new_callable=AsyncMock) as mock_task:
            client.post(
                "/sync",
                content=body,
                headers={"x-hub-signature-256": "sha256=" + "0" * 64,
                         "content-type": "application/json"},
            )
        mock_task.assert_not_called()


# ── Health check still works ─────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

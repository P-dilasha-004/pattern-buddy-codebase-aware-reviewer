"""
Tests for POST /webhook's installation / installation_repositories handling —
the auto-ingest-on-install path that replaces the old manual /admin/reindex
onboarding step. Installing the GitHub App (or granting it more repos) is
now the entire onboarding flow: no PAT, no repo secrets, no manual reindex call.
"""

from __future__ import annotations

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
    import importlib
    import api.main as m
    importlib.reload(m)
    monkeypatch.setattr("api.main._WEBHOOK_SECRET", SECRET)


@pytest.fixture(autouse=True)
def clear_reindex_guard():
    import api.main as m
    m._reindex_in_progress.clear()
    yield
    m._reindex_in_progress.clear()


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app, raise_server_exceptions=True)


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post_webhook(client, payload, event):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "x-hub-signature-256": _sign(body),
            "x-github-event": event,
            "content-type": "application/json",
        },
    )


# ── installation: created ─────────────────────────────────────────────────────

class TestInstallationCreated:
    def test_created_triggers_reindex_for_every_granted_repo(self, client):
        payload = {
            "action": "created",
            "installation": {"id": INSTALLATION_ID},
            "repositories": [{"full_name": "owner/repo-a"}, {"full_name": "owner/repo-b"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "installation")

        assert resp.status_code == 202
        assert mock_task.call_count == 2
        triggered_repos = {call.args[0]["repo_id"] for call in mock_task.call_args_list}
        assert triggered_repos == {"owner/repo-a", "owner/repo-b"}

    def test_triggered_reindex_carries_installation_id(self, client):
        payload = {
            "action": "created",
            "installation": {"id": INSTALLATION_ID},
            "repositories": [{"full_name": "owner/repo-a"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            _post_webhook(client, payload, "installation")

        called_with = mock_task.call_args[0][0]
        assert called_with["installation_id"] == INSTALLATION_ID
        assert called_with["branch"] == "main"

    def test_deleted_action_does_not_trigger_reindex(self, client):
        payload = {
            "action": "deleted",
            "installation": {"id": INSTALLATION_ID},
            "repositories": [{"full_name": "owner/repo-a"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "installation")

        assert resp.status_code == 202
        mock_task.assert_not_called()

    def test_repo_already_reindexing_is_skipped_not_duplicated(self, client):
        import api.main as m
        m._reindex_in_progress.add("owner/repo-a")
        payload = {
            "action": "created",
            "installation": {"id": INSTALLATION_ID},
            "repositories": [{"full_name": "owner/repo-a"}, {"full_name": "owner/repo-b"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            _post_webhook(client, payload, "installation")

        triggered_repos = {call.args[0]["repo_id"] for call in mock_task.call_args_list}
        assert triggered_repos == {"owner/repo-b"}  # repo-a already in flight, skipped


# ── installation_repositories: added/removed ─────────────────────────────────

class TestInstallationRepositoriesChanged:
    def test_added_triggers_reindex(self, client):
        payload = {
            "action": "added",
            "installation": {"id": INSTALLATION_ID},
            "repositories_added": [{"full_name": "owner/new-repo"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "installation_repositories")

        assert resp.status_code == 202
        mock_task.assert_called_once()
        assert mock_task.call_args[0][0]["repo_id"] == "owner/new-repo"

    def test_removed_action_does_not_trigger_reindex(self, client):
        payload = {
            "action": "removed",
            "installation": {"id": INSTALLATION_ID},
            "repositories_removed": [{"full_name": "owner/old-repo"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "installation_repositories")

        assert resp.status_code == 202
        mock_task.assert_not_called()


# ── Missing installation_id ───────────────────────────────────────────────────

class TestMissingInstallationId:
    def test_installation_created_without_id_triggers_nothing(self, client):
        payload = {
            "action": "created",
            "repositories": [{"full_name": "owner/repo-a"}],
        }
        with patch("api.main._process_reindex", new_callable=AsyncMock) as mock_task:
            resp = _post_webhook(client, payload, "installation")

        assert resp.status_code == 202
        mock_task.assert_not_called()

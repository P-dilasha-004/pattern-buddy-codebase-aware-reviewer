"""
Shared fixtures for api/tests/.

Every _process_review / _process_sync / _process_reindex call resolves a
GitHub App installation token before touching the GitHub API (api/main.py's
_resolve_token, via api.github_app). Tests that don't care about GitHub App
auth itself — the vast majority, which mock extract_and_fetch /
post_github_comment / _download_zipball directly — would otherwise hit the
real github_app functions, which need GITHUB_APP_ID/GITHUB_APP_PRIVATE_KEY
and network access, neither of which exist in the test environment.

Patched on api.github_app (the source module), not api.main's imported
names: several test files' own autouse fixtures do
`importlib.reload(api.main)` to pick up a monkeypatched WEBHOOK_SECRET, which
would silently undo a patch applied to api.main's namespace (reload rebinds
every module-level name, including imported ones, back to whatever
api.github_app currently holds). Patching the source module survives that
reload either way, since api.main re-imports from it fresh.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

FAKE_INSTALLATION_TOKEN = "fake-installation-token"
FAKE_INSTALLATION_ID = 999999


@pytest.fixture(autouse=True)
def _default_installation_token(monkeypatch):
    import api.github_app as github_app

    monkeypatch.setattr(
        github_app, "get_installation_token",
        AsyncMock(return_value=FAKE_INSTALLATION_TOKEN),
    )
    monkeypatch.setattr(
        github_app, "list_installations",
        AsyncMock(return_value=[{"id": FAKE_INSTALLATION_ID}]),
    )
    # Any repo_id "matches" the one fake installation — good enough for tests
    # that resolve a token without caring which installation it came from.
    # Tests that care about real installation-membership resolution mock
    # this more precisely within the test body.
    monkeypatch.setattr(
        github_app, "list_installation_repos",
        AsyncMock(side_effect=lambda installation_id, **kw: _AnyRepo()),
    )


class _AnyRepo(list):
    """A list whose `in` check is always True — stands in for "this
    installation covers every repo_id a test might use" without needing to
    enumerate them."""

    def __contains__(self, item):
        return True

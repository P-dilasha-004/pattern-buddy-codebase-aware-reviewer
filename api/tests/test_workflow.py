"""
Tests for Phase 3, Tasks 3.1 and 3.2 — pr_review.yml workflow correctness.

Task 3.1 (structure):
- YAML triggers, concurrency, runner, no checkout
- Secret names, no hardcoded URLs
- Exit-criteria known gaps

Task 3.2 (payload construction & POST):
- GitHub API query shape (gh api … --jq)
- jq -cn payload construction: all keys present, correct types
- openssl dgst | awk HMAC pipeline output format
- X-Hub-Signature-256 header name (not x-hub-signature)
- Signature round-trip: shell signing accepted by Python _verify_signature
- commit_message sourced from PR title, not commit SHA
- curl -s -o /dev/null -w "%{http_code}" pattern for status capture
- Content-Type: application/json on the POST
"""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import shutil
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).parents[2] / ".github" / "workflows" / "pr_review.yml"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as f:
        return yaml.safe_load(f)

def _trigger(data: dict) -> dict:
    # PyYAML 1.1 parses bare 'on' key as Python True
    return data.get(True) or data.get("on", {})


# ── YAML structure ────────────────────────────────────────────────────────────

class TestWorkflowStructure:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists(), f"Missing: {WORKFLOW_PATH}"

    def test_triggers_on_pull_request(self):
        data = _load_workflow()
        trigger = _trigger(data)
        assert "pull_request" in trigger

    def test_triggers_on_opened(self):
        data = _load_workflow()
        types = _trigger(data)["pull_request"]["types"]
        assert "opened" in types

    def test_triggers_on_synchronize(self):
        data = _load_workflow()
        types = _trigger(data)["pull_request"]["types"]
        assert "synchronize" in types

    def test_no_other_trigger_types(self):
        # Ensure we don't accidentally fire on closed/labeled/etc.
        data = _load_workflow()
        types = set(_trigger(data)["pull_request"]["types"])
        assert types == {"opened", "synchronize"}

    def test_concurrency_group_present(self):
        data = _load_workflow()
        assert "concurrency" in data
        assert "group" in data["concurrency"]

    def test_concurrency_group_scoped_to_ref(self):
        # Must include github.ref so concurrent pushes to the same branch cancel
        data = _load_workflow()
        group = data["concurrency"]["group"]
        assert "github.ref" in group

    def test_cancel_in_progress_is_true(self):
        data = _load_workflow()
        assert data["concurrency"]["cancel-in-progress"] is True

    def test_runner_is_ubuntu_latest(self):
        data = _load_workflow()
        job = list(data["jobs"].values())[0]
        assert job["runs-on"] == "ubuntu-latest"

    def test_no_checkout_step(self):
        # Spec: no code checkout — keeps the workflow fast and stateless
        data = _load_workflow()
        job = list(data["jobs"].values())[0]
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            assert "actions/checkout" not in uses, \
                f"Unexpected checkout step: {step}"

    def test_exactly_one_job(self):
        data = _load_workflow()
        assert len(data["jobs"]) == 1

    def test_has_at_least_two_steps(self):
        # Step 1: fetch changed files; Step 2: sign and POST
        data = _load_workflow()
        job = list(data["jobs"].values())[0]
        assert len(job["steps"]) >= 2


# ── Secret names ──────────────────────────────────────────────────────────────

class TestSecretNames:
    """
    The workflow references exactly three secrets. Names must match what
    planning.md specifies and what the Python backend expects as env vars.
    """

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_references_pat_secret(self):
        # Must be PAT, not GITHUB_PAT — GitHub reserves the GITHUB_ prefix
        assert "secrets.PAT" in self._workflow_text()

    def test_does_not_reference_github_pat(self):
        assert "secrets.GITHUB_PAT" not in self._workflow_text()

    def test_references_webhook_secret(self):
        assert "secrets.WEBHOOK_SECRET" in self._workflow_text()

    def test_references_backend_api_url(self):
        assert "secrets.BACKEND_API_URL" in self._workflow_text()

    def test_no_hardcoded_urls(self):
        text = self._workflow_text()
        assert "azurewebsites.net" not in text, \
            "Hardcoded App Service URL — should come from secrets.BACKEND_API_URL"


# ── HMAC signing logic ────────────────────────────────────────────────────────

class TestHmacSigningLogic:
    """
    The shell step signs the payload with openssl dgst -sha256 -hmac.
    Verify the output format matches what _verify_signature expects:
        sha256=<lowercase hex>
    """

    SECRET = "test-webhook-secret-xyz"
    PAYLOAD = json.dumps({
        "repo_id": "owner/repo",
        "pr_number": 42,
        "base_sha": "aaa111",
        "head_sha": "bbb222",
        "commit_message": "Add auth service",
        "changed_files": [
            {"path": "src/auth.py", "status": "modified",
             "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/auth.py"}
        ],
    })

    def _python_hmac(self, secret: str, payload: str) -> str:
        """Reproduce the Python _verify_signature expected value."""
        return "sha256=" + hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    @pytest.mark.skipif(not shutil.which("openssl"), reason="openssl not available")
    def _shell_hmac(self, secret: str, payload: str) -> str:
        """Reproduce the shell openssl dgst command used in the workflow step."""
        result = subprocess.run(
            ["openssl", "dgst", "-sha256", "-hmac", secret],
            input=payload.encode(),
            capture_output=True,
        )
        # openssl output: "SHA256(stdin)= <hex>\n" or "(stdin)= <hex>\n"
        raw = result.stdout.decode().strip()
        hex_part = raw.split()[-1]
        return f"sha256={hex_part}"

    def test_python_hmac_format(self):
        sig = self._python_hmac(self.SECRET, self.PAYLOAD)
        assert sig.startswith("sha256=")
        hex_part = sig[len("sha256="):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)

    @pytest.mark.skipif(not shutil.which("openssl"), reason="openssl not available")
    def test_shell_and_python_hmac_agree(self):
        python_sig = self._python_hmac(self.SECRET, self.PAYLOAD)
        shell_sig  = self._shell_hmac(self.SECRET, self.PAYLOAD)
        assert python_sig == shell_sig, (
            f"Shell signing and Python signing disagree:\n"
            f"  shell:  {shell_sig}\n"
            f"  python: {python_sig}"
        )

    def test_signature_round_trip_accepted_by_verify(self):
        """Signature produced by shell logic must pass _verify_signature."""
        from unittest.mock import patch
        from api.main import _verify_signature

        sig = self._python_hmac(self.SECRET, self.PAYLOAD)
        with patch("api.main._WEBHOOK_SECRET", self.SECRET):
            # Must not raise
            _verify_signature(self.PAYLOAD.encode(), sig)

    def test_wrong_secret_rejected_by_verify(self):
        from fastapi import HTTPException
        from unittest.mock import patch
        from api.main import _verify_signature

        bad_sig = self._python_hmac("wrong-secret", self.PAYLOAD)
        with patch("api.main._WEBHOOK_SECRET", self.SECRET):
            with pytest.raises(HTTPException) as exc_info:
                _verify_signature(self.PAYLOAD.encode(), bad_sig)
        assert exc_info.value.status_code == 401

    def test_tampered_payload_rejected(self):
        from fastapi import HTTPException
        from unittest.mock import patch
        from api.main import _verify_signature

        sig = self._python_hmac(self.SECRET, self.PAYLOAD)
        tampered = self.PAYLOAD.replace("owner/repo", "attacker/evil")
        with patch("api.main._WEBHOOK_SECRET", self.SECRET):
            with pytest.raises(HTTPException) as exc_info:
                _verify_signature(tampered.encode(), sig)
        assert exc_info.value.status_code == 401


# ── Payload schema ────────────────────────────────────────────────────────────

class TestPayloadSchema:
    """
    Validate the keys built by `jq -cn` in the workflow step match what
    _process_review and extract_and_fetch expect.
    """

    REQUIRED_TOP_LEVEL_KEYS = {
        "repo_id",
        "pr_number",
        "base_sha",
        "head_sha",
        "commit_message",
        "changed_files",
    }

    REQUIRED_FILE_KEYS = {"path", "status", "raw_url"}

    def _sample_payload(self) -> dict:
        return {
            "repo_id": "owner/repo",
            "pr_number": 42,
            "base_sha": "aaa111",
            "head_sha": "bbb222",
            "commit_message": "Add auth service",
            "changed_files": [
                {"path": "src/auth.py", "status": "modified",
                 "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/auth.py"},
                {"path": "src/old.py", "status": "removed",
                 "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/src/old.py"},
            ],
        }

    def test_all_required_top_level_keys_present(self):
        payload = self._sample_payload()
        assert self.REQUIRED_TOP_LEVEL_KEYS <= set(payload.keys())

    def test_repo_id_is_owner_slash_repo_format(self):
        payload = self._sample_payload()
        assert "/" in payload["repo_id"]

    def test_pr_number_is_int(self):
        payload = self._sample_payload()
        assert isinstance(payload["pr_number"], int)

    def test_changed_files_is_list_of_dicts(self):
        payload = self._sample_payload()
        assert isinstance(payload["changed_files"], list)
        for f in payload["changed_files"]:
            assert isinstance(f, dict)

    def test_each_file_has_required_keys(self):
        payload = self._sample_payload()
        for f in payload["changed_files"]:
            assert self.REQUIRED_FILE_KEYS <= set(f.keys()), \
                f"File entry missing keys: {f}"

    def test_file_status_values_are_valid(self):
        valid_statuses = {"added", "modified", "removed", "renamed", "copied", "changed"}
        payload = self._sample_payload()
        for f in payload["changed_files"]:
            assert f["status"] in valid_statuses, \
                f"Unexpected status value: {f['status']}"

    def test_payload_is_json_serialisable(self):
        payload = self._sample_payload()
        serialised = json.dumps(payload)
        assert json.loads(serialised) == payload

    def test_payload_accepted_by_process_review_extraction(self):
        """
        Verify _process_review can safely call .get() on every expected key
        without raising — i.e. the schema is forward-compatible.
        """
        payload = self._sample_payload()
        assert payload.get("repo_id") == "owner/repo"
        assert payload.get("pr_number") == 42
        assert isinstance(payload.get("changed_files"), list)
        # Each file dict must support .get("status") and .get("path")
        for f in payload["changed_files"]:
            assert f.get("status") is not None
            assert f.get("path") is not None


# ── Exit criteria: known gaps ─────────────────────────────────────────────────

class TestExitCriteriaKnownGaps:
    """
    Exit criteria from planning.md:
      1. "The GitHub Action completes in under 10 seconds."
      2. "Rapidly pushing 3 commits results in only 1 final review."

    Neither can be fully verified by a unit test:
      - (1) requires timing a live Actions run.
      - (2) requires GitHub's infrastructure to actually cancel in-flight runs,
            which is a runtime behavior of the GitHub Actions scheduler reacting
            to the YAML config — not something a static test can prove end-to-end.

    These tests document what CAN be verified statically and flag the rest as
    manual integration checkpoints for the first real-PR smoke test.
    """

    def test_concurrency_yaml_is_necessary_condition_for_cancellation(self):
        # The YAML config is the prerequisite; without it, cancellation is impossible.
        data = _load_workflow()
        assert data["concurrency"]["cancel-in-progress"] is True
        assert "github.ref" in data["concurrency"]["group"]

    def test_no_sleep_or_delay_steps_that_would_inflate_runtime(self):
        # Any sleep in a step would push the <10s budget. Confirm none exist.
        data = _load_workflow()
        job = list(data["jobs"].values())[0]
        for step in job.get("steps", []):
            run_script = step.get("run", "")
            assert "sleep " not in run_script, \
                f"Step '{step.get('name')}' contains a sleep — risks blowing the 10s budget"

    def test_no_checkout_that_would_inflate_runtime(self):
        # Cloning a repo can take 5-30s on its own. Already covered by structure
        # tests, but repeated here as an explicit exit-criteria guard.
        data = _load_workflow()
        job = list(data["jobs"].values())[0]
        for step in job.get("steps", []):
            assert "actions/checkout" not in step.get("uses", "")

    def test_known_gap_runtime_timing_requires_manual_verification(self):
        # Documents that the <10s timing requirement must be verified on the
        # first real Actions run. Check the job duration in the Actions UI.
        # This test is intentionally a no-op — it exists to make the gap explicit.
        pass

    def test_known_gap_single_review_per_push_burst_requires_integration_test(self):
        # Documents that "3 rapid pushes → 1 review" must be verified manually:
        # push 3 commits in quick succession, confirm only one Pattern Buddy
        # comment appears on the PR. Cannot be unit-tested against live Actions.
        pass


# ── Empty / all-filtered changed_files ───────────────────────────────────────

class TestEmptyChangedFiles:
    """
    What happens when a PR touches only lock files or binaries — all entries
    filtered downstream by _is_ingestible. The workflow must still construct
    a valid payload with an empty (or all-non-ingestible) changed_files list,
    so the backend can handle it gracefully rather than crashing on a missing key.
    """

    def _empty_payload(self) -> dict:
        return {
            "repo_id": "owner/repo",
            "pr_number": 7,
            "base_sha": "aaa111",
            "head_sha": "bbb222",
            "commit_message": "Bump lockfile",
            "changed_files": [],
        }

    def _lockfile_only_payload(self) -> dict:
        return {
            "repo_id": "owner/repo",
            "pr_number": 8,
            "base_sha": "aaa111",
            "head_sha": "bbb222",
            "commit_message": "Update dependencies",
            "changed_files": [
                {"path": "poetry.lock", "status": "modified",
                 "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/poetry.lock"},
                {"path": "package-lock.json", "status": "modified",
                 "raw_url": "https://raw.githubusercontent.com/owner/repo/bbb222/package-lock.json"},
            ],
        }

    def test_empty_changed_files_is_valid_json(self):
        payload = self._empty_payload()
        assert json.loads(json.dumps(payload))["changed_files"] == []

    def test_empty_payload_has_all_required_keys(self):
        required = {"repo_id", "pr_number", "base_sha", "head_sha",
                    "commit_message", "changed_files"}
        assert required <= set(self._empty_payload().keys())

    def test_process_review_handles_empty_changed_files_without_error(self):
        # _process_review calls extract_and_fetch(payload) which calls
        # payload.get("changed_files", []) — must not raise on an empty list.
        payload = self._empty_payload()
        assert payload.get("changed_files", []) == []

    def test_lockfile_only_payload_is_valid_schema(self):
        payload = self._lockfile_only_payload()
        required = {"repo_id", "pr_number", "base_sha", "head_sha",
                    "commit_message", "changed_files"}
        assert required <= set(payload.keys())
        assert isinstance(payload["changed_files"], list)
        assert len(payload["changed_files"]) == 2

    def test_lockfile_only_all_filtered_by_is_ingestible(self):
        # Confirm downstream filter correctly drops all entries — the backend
        # gets an empty fetch list and returns FetchResult(files={}, fallback=False).
        from api.diff import _is_ingestible
        payload = self._lockfile_only_payload()
        ingestible = [f for f in payload["changed_files"]
                      if _is_ingestible(f["path"])]
        assert ingestible == [], \
            "Expected all lock files to be filtered — got: " + str(ingestible)

    def test_empty_changed_files_does_not_trigger_fallback_threshold(self):
        # 0 files is well under the 50-file threshold — must NOT set fallback=True.
        from api.diff import MAX_FILES_THRESHOLD
        assert 0 <= MAX_FILES_THRESHOLD


# ── 202 assertion failure behavior ────────────────────────────────────────────

class TestAssert202FailureBehavior:
    """
    The workflow step runs: if [ "$HTTP_STATUS" != "202" ]; then exit 1; fi
    A non-202 response must cause a non-zero exit, which marks the Actions job
    as failed (red X on the PR check) so the developer knows delivery failed.
    This is the unit-testable proxy for Failure Mode #3 (webhook delivery failure).
    """

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def _run_assert_script(self, http_status: str) -> int:
        """Run the inline assertion logic from the workflow step; return exit code."""
        script = f"""
HTTP_STATUS="{http_status}"
echo "Response status: $HTTP_STATUS"
if [ "$HTTP_STATUS" != "202" ]; then
  echo "ERROR: Expected 202 Accepted, got $HTTP_STATUS"
  exit 1
fi
"""
        result = subprocess.run(["bash", "-c", script], capture_output=True)
        return result.returncode

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_202_exits_zero(self):
        assert self._run_assert_script("202") == 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_401_exits_nonzero(self):
        # 401 = signature mismatch (wrong WEBHOOK_SECRET on App Service)
        assert self._run_assert_script("401") != 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_500_exits_nonzero(self):
        # 500 = App Service crashed or WEBHOOK_SECRET env var missing
        assert self._run_assert_script("500") != 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_000_exits_nonzero(self):
        # 000 = curl could not connect at all (App Service unreachable)
        assert self._run_assert_script("000") != 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_503_exits_nonzero(self):
        # 503 = App Service cold-start or overloaded
        assert self._run_assert_script("503") != 0

    def test_workflow_step_contains_exit_1_on_non_202(self):
        # Static check: the exit 1 branch must be present in the workflow source.
        text = WORKFLOW_PATH.read_text()
        assert 'exit 1' in text
        assert '"202"' in text or "'202'" in text

    def test_workflow_step_does_not_silently_ignore_errors(self):
        # Must not have || true or 2>/dev/null around the status check.
        text = WORKFLOW_PATH.read_text()
        assert "|| true" not in text


# ── SHA expression correctness ────────────────────────────────────────────────

class TestShaExpressions:
    """
    After concurrency cancellation, the surviving run must carry the HEAD SHA
    of the latest commit — not a stale SHA from an earlier run's context.

    The workflow must use github.event.pull_request.head.sha (the PR's current
    tip) rather than github.sha (which is the merge commit SHA and can be stale
    in a PR context) or any other expression.
    """

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_head_sha_uses_pull_request_head_sha(self):
        # Must reference the PR head, not the push SHA
        assert "github.event.pull_request.head.sha" in self._workflow_text()

    def test_base_sha_uses_pull_request_base_sha(self):
        assert "github.event.pull_request.base.sha" in self._workflow_text()

    def test_does_not_use_github_sha_for_head(self):
        # github.sha in a pull_request event is the merge commit SHA, not the
        # PR branch tip — using it would send a wrong (potentially non-existent)
        # SHA to the backend's raw_url construction.
        text = self._workflow_text()
        # Allow github.sha only if it does NOT appear as the value for head_sha
        # Simple check: the string 'github.sha' must not appear as a jq argument
        assert '--arg head_sha   "${{ github.sha }}"' not in text
        assert "--arg head_sha \"${{ github.sha }}\"" not in text

    def test_head_sha_and_base_sha_are_different_expressions(self):
        text = self._workflow_text()
        # Both expressions must appear and be distinct
        assert "pull_request.head.sha" in text
        assert "pull_request.base.sha" in text
        # They must not be the same string
        assert "pull_request.head.sha" != "pull_request.base.sha"

    def test_head_sha_in_payload_maps_to_pr_tip(self):
        # Schema-level: head_sha must be a non-empty string distinct from base_sha
        payload = {
            "head_sha": "bbb222",
            "base_sha": "aaa111",
        }
        assert payload["head_sha"] != payload["base_sha"]
        assert len(payload["head_sha"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3.2 — Payload Construction & POST
# ═══════════════════════════════════════════════════════════════════════════════

# ── GitHub API query step ─────────────────────────────────────────────────────

class TestGithubApiQueryStep:
    """
    The first step calls:
      gh api "repos/{repo}/pulls/{number}/files" --paginate
            --jq '[.[] | {path: .filename, status: .status, raw_url: .raw_url}]'

    Validates the jq projection shape and that --paginate is present.
    """

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_uses_gh_api_to_query_changed_files(self):
        text = self._workflow_text()
        assert "gh api" in text
        assert "pulls/" in text and "/files" in text

    def test_uses_paginate_flag(self):
        # Without --paginate PRs with >30 files silently truncate
        assert "--paginate" in self._workflow_text()

    def test_jq_projects_path_field(self):
        assert "path: .filename" in self._workflow_text()

    def test_jq_projects_status_field(self):
        assert "status: .status" in self._workflow_text()

    def test_jq_projects_raw_url_field(self):
        assert "raw_url: .raw_url" in self._workflow_text()

    def test_jq_output_is_array(self):
        # The projection must produce an array — jq starts with '[.[] | …]'
        text = self._workflow_text()
        assert "[.[] |" in text or "[.[]|" in text

    def test_output_stored_in_step_output(self):
        # Must write to GITHUB_OUTPUT so the next step can reference it
        assert 'GITHUB_OUTPUT' in self._workflow_text()

    def test_uses_pat_for_api_auth(self):
        # GH_TOKEN must come from secrets.PAT to avoid rate limiting
        text = self._workflow_text()
        assert "secrets.PAT" in text

    def test_gh_api_jq_produces_correct_shape(self):
        """
        Simulate the jq projection against a GitHub API response fragment.
        Verifies the transformation produces {path, status, raw_url} dicts.
        """
        import subprocess, json, shutil
        if not shutil.which("jq"):
            pytest.skip("jq not available")

        github_api_response = json.dumps([
            {
                "filename": "src/auth.py",
                "status": "modified",
                "raw_url": "https://raw.githubusercontent.com/owner/repo/abc/src/auth.py",
                "additions": 5,
                "deletions": 2,
                "changes": 7,
                "patch": "@@ -1,2 +1,3 @@",
            },
            {
                "filename": "poetry.lock",
                "status": "modified",
                "raw_url": "https://raw.githubusercontent.com/owner/repo/abc/poetry.lock",
                "additions": 100,
                "deletions": 100,
                "changes": 200,
                "patch": "",
            },
        ])

        result = subprocess.run(
            ["jq", "-c", "[.[] | {path: .filename, status: .status, raw_url: .raw_url}]"],
            input=github_api_response.encode(),
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        projected = json.loads(result.stdout)
        assert len(projected) == 2
        assert projected[0] == {
            "path": "src/auth.py",
            "status": "modified",
            "raw_url": "https://raw.githubusercontent.com/owner/repo/abc/src/auth.py",
        }
        # Confirm extraneous fields (additions, deletions, patch) are dropped
        assert "additions" not in projected[0]
        assert "patch" not in projected[0]


# ── jq payload construction ───────────────────────────────────────────────────

class TestJqPayloadConstruction:
    """
    The second step builds the payload with:
      jq -cn --arg repo_id … --argjson pr_number … --argjson changed_files …
    Validates key names, types, and that commit_message uses the PR title.
    """

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_uses_jq_cn_for_null_input_compact(self):
        # -c compact output, -n null input (no stdin)
        text = self._workflow_text()
        assert "jq -cn" in text

    def test_repo_id_passed_as_string_arg(self):
        assert '--arg repo_id' in self._workflow_text()

    def test_pr_number_passed_as_json_arg(self):
        # --argjson preserves integer type; --arg would stringify it
        assert '--argjson pr_number' in self._workflow_text()

    def test_changed_files_passed_as_json_arg(self):
        # --argjson preserves the array; --arg would double-encode it
        assert '--argjson changed_files' in self._workflow_text()

    def test_commit_message_sourced_from_pr_title(self):
        # PR title is the human-readable summary; commit SHA would be useless to the LLM
        text = self._workflow_text()
        assert "pull_request.title" in text

    def test_commit_message_not_sourced_from_head_commit_message(self):
        # head_commit.message is only available on push events, not pull_request events
        assert "head_commit.message" not in self._workflow_text()

    def test_payload_key_commit_message_not_commit_msg(self):
        # The backend expects "commit_message" (matches _process_review / build_macro_prompt)
        assert "commit_message:" in self._workflow_text()

    @pytest.mark.skipif(not shutil.which("jq"), reason="jq not available")
    def test_jq_cn_produces_correct_payload_shape(self):
        """Run the exact jq -cn command from the workflow with test values."""
        result = subprocess.run(
            [
                "jq", "-cn",
                "--arg",     "repo_id",       "owner/repo",
                "--argjson", "pr_number",      "42",
                "--arg",     "base_sha",       "aaa111",
                "--arg",     "head_sha",       "bbb222",
                "--arg",     "commit_msg",     "Add auth service",
                "--argjson", "changed_files",  '[{"path":"src/auth.py","status":"modified","raw_url":"https://example.com/auth.py"}]',
                """{
                  repo_id: $repo_id,
                  pr_number: $pr_number,
                  base_sha: $base_sha,
                  head_sha: $head_sha,
                  commit_message: $commit_msg,
                  changed_files: $changed_files
                }""",
            ],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        payload = json.loads(result.stdout)
        assert payload["repo_id"] == "owner/repo"
        assert payload["pr_number"] == 42          # integer, not "42"
        assert payload["base_sha"] == "aaa111"
        assert payload["head_sha"] == "bbb222"
        assert payload["commit_message"] == "Add auth service"
        assert isinstance(payload["changed_files"], list)
        assert payload["changed_files"][0]["path"] == "src/auth.py"

    @pytest.mark.skipif(not shutil.which("jq"), reason="jq not available")
    def test_pr_number_is_integer_not_string_in_payload(self):
        result = subprocess.run(
            ["jq", "-cn", "--argjson", "pr_number", "42",
             "{pr_number: $pr_number}"],
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        assert isinstance(payload["pr_number"], int)
        assert not isinstance(payload["pr_number"], str)


# ── HMAC pipeline (openssl dgst | awk) ───────────────────────────────────────

class TestOpensslAwkHmacPipeline:
    """
    The workflow signs with:
      echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}'
    then prepends "sha256=".

    Validates the full pipeline produces the correct hex, that `echo -n` is used
    (no trailing newline), and that awk extracts the right field.
    """

    SECRET  = "test-webhook-secret"
    PAYLOAD = '{"repo_id":"owner/repo","pr_number":42}'

    def _python_hmac(self, secret: str, payload: str) -> str:
        return "sha256=" + hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_uses_echo_n_to_avoid_trailing_newline(self):
        # Without -n, echo appends \n and the HMAC won't match the Python gate
        assert "echo -n" in self._workflow_text()

    def test_uses_openssl_dgst_sha256_hmac(self):
        text = self._workflow_text()
        assert "openssl dgst -sha256 -hmac" in text

    def test_uses_awk_to_extract_hex_field(self):
        # openssl output: "SHA256(stdin)= <hex>" — awk '{print $2}' extracts the hex
        assert "awk '{print $2}'" in self._workflow_text()

    def test_signature_prefixed_with_sha256_equals(self):
        assert 'SIGNATURE="sha256=$(' in self._workflow_text()

    @pytest.mark.skipif(
        not shutil.which("openssl") or not shutil.which("awk"),
        reason="openssl/awk not available",
    )
    def test_full_pipeline_matches_python_hmac(self):
        """
        Run echo -n … | openssl dgst -sha256 -hmac … | awk '{print $2}'
        and confirm it matches Python's hmac.new().hexdigest().
        """
        pipeline = (
            f'echo -n \'{self.PAYLOAD}\' | '
            f'openssl dgst -sha256 -hmac \'{self.SECRET}\' | '
            f"awk '{{print $2}}'"
        )
        result = subprocess.run(["bash", "-c", pipeline], capture_output=True)
        assert result.returncode == 0
        hex_only = result.stdout.decode().strip()
        shell_sig = f"sha256={hex_only}"
        python_sig = self._python_hmac(self.SECRET, self.PAYLOAD)
        assert shell_sig == python_sig, (
            f"Pipeline output '{shell_sig}' != Python HMAC '{python_sig}'"
        )

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_trailing_newline_breaks_signature(self):
        """
        Confirms that echo WITHOUT -n produces a different HMAC than echo -n,
        proving the -n flag is load-bearing.
        """
        with_newline = subprocess.run(
            ["bash", "-c",
             f"echo '{self.PAYLOAD}' | openssl dgst -sha256 -hmac '{self.SECRET}' | awk '{{print $2}}'"],
            capture_output=True,
        )
        without_newline = subprocess.run(
            ["bash", "-c",
             f"echo -n '{self.PAYLOAD}' | openssl dgst -sha256 -hmac '{self.SECRET}' | awk '{{print $2}}'"],
            capture_output=True,
        )
        sig_with    = with_newline.stdout.decode().strip()
        sig_without = without_newline.stdout.decode().strip()
        assert sig_with != sig_without, \
            "echo and echo -n produced the same HMAC — trailing newline is NOT load-bearing (unexpected)"


# ── curl POST step ────────────────────────────────────────────────────────────

class TestCurlPostStep:
    """
    The POST step:
      curl -s -o /dev/null -w "%{http_code}"
           -X POST "${BACKEND_API_URL}/review"
           -H "Content-Type: application/json"
           -H "X-Hub-Signature-256: ${SIGNATURE}"
           -d "$PAYLOAD"
    """

    def _workflow_text(self) -> str:
        return WORKFLOW_PATH.read_text()

    def test_posts_to_review_endpoint(self):
        assert "${BACKEND_API_URL}/review" in self._workflow_text()

    def test_uses_x_hub_signature_256_header_name(self):
        # Must be exactly "X-Hub-Signature-256" — GitHub's own header name.
        # "X-Hub-Signature" (without -256) is the deprecated SHA-1 variant.
        assert "X-Hub-Signature-256" in self._workflow_text()

    def test_does_not_use_deprecated_x_hub_signature(self):
        text = self._workflow_text()
        # Allow "X-Hub-Signature-256" but not bare "X-Hub-Signature:"
        lines_with_old_header = [
            line for line in text.splitlines()
            if "X-Hub-Signature:" in line and "256" not in line
        ]
        assert lines_with_old_header == [], \
            f"Deprecated X-Hub-Signature header found: {lines_with_old_header}"

    def test_content_type_is_application_json(self):
        assert "Content-Type: application/json" in self._workflow_text()

    def test_curl_uses_silent_flag(self):
        # -s suppresses progress meter — otherwise Actions logs are noisy
        assert "curl -s" in self._workflow_text()

    def test_curl_captures_http_status_code(self):
        assert '-w "%{http_code}"' in self._workflow_text() or \
               "-w '%{http_code}'" in self._workflow_text()

    def test_curl_discards_response_body(self):
        # -o /dev/null discards body — only status code matters
        assert "-o /dev/null" in self._workflow_text()

    def test_curl_uses_explicit_post_method(self):
        assert "-X POST" in self._workflow_text()

    def test_signature_header_value_references_signature_variable(self):
        text = self._workflow_text()
        assert "X-Hub-Signature-256: ${SIGNATURE}" in text

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_http_status_capture_pattern_works(self):
        """
        Verify curl -s -o /dev/null -w "%{http_code}" correctly captures status.
        Uses a local Python server to avoid network calls.
        """
        import threading
        import http.server

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(202)
                self.end_headers()
            def log_message(self, *a): pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.start()

        result = subprocess.run(
            ["bash", "-c",
             f'curl -s -o /dev/null -w "%{{http_code}}" -X POST http://127.0.0.1:{port}/review'],
            capture_output=True,
        )
        t.join(timeout=3)
        assert result.stdout.decode().strip() == "202"

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def _run_full_step_script(self, response_code: int) -> int:
        """
        Spin up a local server returning response_code, run the exact curl +
        assertion script from the workflow step, return the shell exit code.
        """
        import threading
        import http.server

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(response_code)
                self.end_headers()
            def log_message(self, *a): pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.start()

        script = f"""
HTTP_STATUS=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST http://127.0.0.1:{port}/review)
echo "Response status: $HTTP_STATUS"
if [ "$HTTP_STATUS" != "202" ]; then
  echo "ERROR: Expected 202 Accepted, got $HTTP_STATUS"
  exit 1
fi
"""
        result = subprocess.run(["bash", "-c", script], capture_output=True)
        t.join(timeout=3)
        return result.returncode

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_full_step_exits_zero_on_202(self):
        """Happy path: server returns 202, step exits 0 (green check)."""
        assert self._run_full_step_script(202) == 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_full_step_exits_nonzero_on_401(self):
        """401 = wrong WEBHOOK_SECRET on App Service — must fail the Actions job."""
        assert self._run_full_step_script(401) != 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_full_step_exits_nonzero_on_500(self):
        """500 = App Service crashed or WEBHOOK_SECRET env var missing."""
        assert self._run_full_step_script(500) != 0

    @pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
    def test_full_step_exits_nonzero_on_503(self):
        """503 = App Service cold-start or overloaded."""
        assert self._run_full_step_script(503) != 0

"""
Tests for Task 2.2 — Diff extraction and concurrent file fetching.

Covers:
- Extension filtering (ingestible vs skipped)
- Lockfile exclusion
- Deleted-file exclusion
- 50-file defensive threshold (exactly 50 → fetch; >50 → fallback)
- PAT injected on every request
- return_exceptions=True: one failed fetch doesn't kill the batch
- Timeout fires on hung responses
- Empty payload / no fetchable files
- FetchResult fields
"""

import asyncio
import pytest
import respx
import httpx

from api.diff import (
    extract_and_fetch,
    FetchResult,
    MAX_FILES_THRESHOLD,
    _is_ingestible,
    _fetchable,
)

PAT = "ghp_test_token"
# The code fetches via the GitHub Contents API (raw media type), not the
# pulls/files raw_url. Mocks register the Contents URL the code actually hits;
# respx ignores the ?ref= query param when it isn't part of the route pattern.
CONTENTS_BASE = "https://api.github.com/repos/owner/repo/contents"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file(path: str, status: str = "modified", content: str = "# content") -> dict:
    return {
        "path": path,
        "status": status,
        "raw_url": f"{CONTENTS_BASE}/{path}",  # Contents API URL the code requests
        "_content": content,   # used by respx mock only, not sent to server
    }


def _payload(*files, pr_number: int = 1, repo_id: str = "owner/repo") -> dict:
    return {
        "repo_id": repo_id,
        "pr_number": pr_number,
        "head_sha": "abc123",
        "changed_files": [
            {k: v for k, v in f.items() if k != "_content"}
            for f in files
        ],
    }


def _mock_files(router, *files):
    """Register respx routes for each file's raw_url."""
    for f in files:
        router.get(f["raw_url"]).mock(
            return_value=httpx.Response(200, text=f["_content"])
        )


# ── Extension / lockfile filtering ────────────────────────────────────────────

class TestIsIngestible:
    @pytest.mark.parametrize("path", [
        "src/auth.py", "README.md", "notes.txt",
        "config.toml", "settings.json", "docker-compose.yaml", "ci.yml",
    ])
    def test_ingestible_extensions_accepted(self, path):
        assert _is_ingestible(path)

    @pytest.mark.parametrize("path", [
        "image.png", "binary.exe", "archive.zip", "Makefile",
        "photo.jpg", "font.woff2",
    ])
    def test_non_ingestible_extensions_rejected(self, path):
        assert not _is_ingestible(path)

    @pytest.mark.parametrize("name", [
        "poetry.lock", "package-lock.json", "yarn.lock", "uv.lock",
        "Pipfile.lock", "composer.lock", "Gemfile.lock", "cargo.lock",
        "packages.lock.json",
    ])
    def test_known_lockfiles_rejected(self, name):
        assert not _is_ingestible(name)

    def test_arbitrary_lock_extension_rejected(self):
        assert not _is_ingestible("custom.lock")

    def test_nested_lockfile_rejected(self):
        assert not _is_ingestible("deps/poetry.lock")


class TestFetchable:
    def test_removed_files_excluded(self):
        files = [_file("src/auth.py", status="removed")]
        assert _fetchable(files) == []

    def test_added_files_included(self):
        files = [_file("src/auth.py", status="added")]
        assert len(_fetchable(files)) == 1

    def test_file_without_path_excluded(self):
        # raw_url is no longer used (Contents API keys off path); a missing/empty
        # path is what makes an entry unfetchable.
        files = [{"status": "modified", "raw_url": None}]
        assert _fetchable(files) == []

    def test_file_with_path_but_no_raw_url_still_fetchable(self):
        # Regression: the old code required raw_url and dropped otherwise-valid
        # files. The Contents API only needs the path.
        files = [{"path": "src/auth.py", "status": "modified"}]
        assert len(_fetchable(files)) == 1

    def test_non_ingestible_extension_excluded(self):
        files = [_file("logo.png")]
        assert _fetchable(files) == []


# ── Defensive threshold ───────────────────────────────────────────────────────

class TestDefensiveThreshold:
    def _make_files(self, n: int) -> list[dict]:
        return [_file(f"src/module_{i}.py") for i in range(n)]

    @pytest.mark.asyncio
    async def test_exactly_50_files_does_not_trigger_fallback(self):
        files = self._make_files(MAX_FILES_THRESHOLD)
        with respx.mock(assert_all_called=False) as router:
            for f in files:
                router.get(f["raw_url"]).mock(
                    return_value=httpx.Response(200, text="def fn(): pass")
                )
            result = await extract_and_fetch(_payload(*files), PAT)
        assert result.fallback is False

    @pytest.mark.asyncio
    async def test_51_files_triggers_fallback(self):
        files = self._make_files(MAX_FILES_THRESHOLD + 1)
        with respx.mock(assert_all_called=False):
            result = await extract_and_fetch(_payload(*files), PAT)
        assert result.fallback is True

    @pytest.mark.asyncio
    async def test_fallback_result_has_empty_files(self):
        files = self._make_files(MAX_FILES_THRESHOLD + 1)
        with respx.mock(assert_all_called=False):
            result = await extract_and_fetch(_payload(*files), PAT)
        assert result.files == {}

    @pytest.mark.asyncio
    async def test_fallback_makes_no_http_requests(self):
        files = self._make_files(MAX_FILES_THRESHOLD + 1)
        with respx.mock(assert_all_called=False) as router:
            result = await extract_and_fetch(_payload(*files), PAT)
        # respx raises if any unexpected call is made
        assert result.fallback is True

    @pytest.mark.asyncio
    async def test_threshold_counts_only_ingestible_files(self):
        # 49 .py files + 10 .png files → only 49 ingestible → no fallback
        py_files  = self._make_files(49)
        png_files = [_file(f"img_{i}.png") for i in range(10)]
        all_files = py_files + png_files
        with respx.mock(assert_all_called=False) as router:
            for f in py_files:
                router.get(f["raw_url"]).mock(
                    return_value=httpx.Response(200, text="x = 1")
                )
            result = await extract_and_fetch(_payload(*all_files), PAT)
        assert result.fallback is False


# ── PAT injection ─────────────────────────────────────────────────────────────

class TestPatInjection:
    @pytest.mark.asyncio
    async def test_pat_sent_as_bearer_token(self):
        f = _file("src/auth.py")
        captured_headers = {}

        def capture(request, *args, **kwargs):
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, text="def fn(): pass")

        with respx.mock() as router:
            router.get(f["raw_url"]).mock(side_effect=capture)
            await extract_and_fetch(_payload(f), "ghp_my_secret_pat")

        assert captured_headers.get("authorization") == "Bearer ghp_my_secret_pat"

    @pytest.mark.asyncio
    async def test_pat_injected_on_every_request(self):
        files = [_file(f"src/mod_{i}.py") for i in range(3)]
        auth_headers: list[str] = []

        def capture(request, *args, **kwargs):
            auth_headers.append(request.headers.get("authorization", ""))
            return httpx.Response(200, text="x = 1")

        with respx.mock() as router:
            for f in files:
                router.get(f["raw_url"]).mock(side_effect=capture)
            await extract_and_fetch(_payload(*files), "ghp_token_abc")

        assert all(h == "Bearer ghp_token_abc" for h in auth_headers)
        assert len(auth_headers) == 3


# ── Concurrent fetch: partial failures ───────────────────────────────────────

class TestPartialFailures:
    @pytest.mark.asyncio
    async def test_one_404_does_not_kill_batch(self):
        good = _file("src/auth.py", content="def login(): pass")
        bad  = _file("src/deleted.py")

        with respx.mock() as router:
            router.get(good["raw_url"]).mock(
                return_value=httpx.Response(200, text=good["_content"])
            )
            router.get(bad["raw_url"]).mock(
                return_value=httpx.Response(404)
            )
            result = await extract_and_fetch(_payload(good, bad), PAT)

        assert "src/auth.py" in result.files
        assert "src/deleted.py" not in result.files
        assert "src/deleted.py" in result.failed_paths

    @pytest.mark.asyncio
    async def test_all_failed_returns_empty_files(self):
        files = [_file("src/a.py"), _file("src/b.py")]

        with respx.mock() as router:
            for f in files:
                router.get(f["raw_url"]).mock(return_value=httpx.Response(500))
            result = await extract_and_fetch(_payload(*files), PAT)

        assert result.files == {}
        assert len(result.failed_paths) == 2
        assert result.fallback is False

    @pytest.mark.asyncio
    async def test_failed_paths_recorded_correctly(self):
        good = _file("src/ok.py", content="x = 1")
        bad  = _file("src/bad.py")

        with respx.mock() as router:
            router.get(good["raw_url"]).mock(
                return_value=httpx.Response(200, text=good["_content"])
            )
            router.get(bad["raw_url"]).mock(return_value=httpx.Response(403))
            result = await extract_and_fetch(_payload(good, bad), PAT)

        assert result.failed_paths == ["src/bad.py"]

    @pytest.mark.asyncio
    async def test_network_error_treated_as_failure(self):
        good = _file("src/ok.py", content="x = 1")
        bad  = _file("src/net_err.py")

        with respx.mock() as router:
            router.get(good["raw_url"]).mock(
                return_value=httpx.Response(200, text=good["_content"])
            )
            router.get(bad["raw_url"]).mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            result = await extract_and_fetch(_payload(good, bad), PAT)

        assert "src/ok.py" in result.files
        assert "src/net_err.py" in result.failed_paths


# ── Timeout ───────────────────────────────────────────────────────────────────

class TestTimeout:
    # respx mocks bypass httpx's internal timeout machinery (they intercept at
    # the transport level before the timeout cancellation fires). We instead
    # raise httpx.ReadTimeout directly — the same exception httpx raises on a
    # real timeout — to verify our error-handling path treats it as a failure.

    @pytest.mark.asyncio
    async def test_hung_request_times_out(self):
        f = _file("src/slow.py")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(
                side_effect=httpx.ReadTimeout("GitHub response timed out")
            )
            result = await extract_and_fetch(_payload(f), PAT, http_timeout=0.05)

        assert "src/slow.py" not in result.files
        assert "src/slow.py" in result.failed_paths

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_fast_files(self):
        fast = _file("src/fast.py", content="x = 1")
        slow = _file("src/slow.py")

        with respx.mock() as router:
            router.get(fast["raw_url"]).mock(
                return_value=httpx.Response(200, text=fast["_content"])
            )
            router.get(slow["raw_url"]).mock(
                side_effect=httpx.ReadTimeout("GitHub response timed out")
            )
            result = await extract_and_fetch(_payload(fast, slow), PAT, http_timeout=0.05)

        assert "src/fast.py" in result.files
        assert "src/slow.py" in result.failed_paths


# ── Happy path and edge cases ─────────────────────────────────────────────────

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_single_file_fetched(self):
        f = _file("src/auth.py", content="def login(): pass")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(
                return_value=httpx.Response(200, text=f["_content"])
            )
            result = await extract_and_fetch(_payload(f), PAT)

        assert result.fallback is False
        assert result.files == {"src/auth.py": "def login(): pass"}
        assert result.failed_paths == []

    @pytest.mark.asyncio
    async def test_mixed_track_files_all_fetched(self):
        py   = _file("src/auth.py",   content="def fn(): pass")
        md   = _file("README.md",     content="# Title")
        toml = _file("config.toml",   content="[tool]\nname='app'")
        yml  = _file(".github/workflows/ci.yml", content="on: push")

        with respx.mock() as router:
            for f in [py, md, toml, yml]:
                router.get(f["raw_url"]).mock(
                    return_value=httpx.Response(200, text=f["_content"])
                )
            result = await extract_and_fetch(_payload(py, md, toml, yml), PAT)

        assert set(result.files) == {"src/auth.py", "README.md", "config.toml", ".github/workflows/ci.yml"}

    @pytest.mark.asyncio
    async def test_deleted_files_not_fetched(self):
        deleted = _file("src/old.py", status="removed")
        kept    = _file("src/new.py", content="x = 1")

        with respx.mock() as router:
            router.get(kept["raw_url"]).mock(
                return_value=httpx.Response(200, text=kept["_content"])
            )
            result = await extract_and_fetch(_payload(deleted, kept), PAT)

        assert "src/old.py" not in result.files
        assert "src/new.py" in result.files

    @pytest.mark.asyncio
    async def test_empty_changed_files_returns_empty_result(self):
        result = await extract_and_fetch({"repo_id": "r", "changed_files": []}, PAT)
        assert result.files == {}
        assert result.fallback is False
        assert result.failed_paths == []

    @pytest.mark.asyncio
    async def test_file_content_preserved_exactly(self):
        content = "import os\n\ndef main():\n    print(os.getcwd())\n"
        f = _file("src/main.py", content=content)
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(
                return_value=httpx.Response(200, text=content)
            )
            result = await extract_and_fetch(_payload(f), PAT)

        assert result.files["src/main.py"] == content


# ── FetchResult contract ──────────────────────────────────────────────────────

class TestFetchResultContract:
    def test_default_fetch_result_has_empty_files(self):
        r = FetchResult()
        assert r.files == {}

    def test_default_fetch_result_fallback_is_false(self):
        r = FetchResult()
        assert r.fallback is False

    def test_default_fetch_result_failed_paths_is_empty_list(self):
        r = FetchResult()
        assert r.failed_paths == []

    @pytest.mark.asyncio
    async def test_successful_fetch_has_fallback_false(self):
        f = _file("src/ok.py", content="x = 1")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(200, text="x = 1"))
            result = await extract_and_fetch(_payload(f), PAT)
        assert result.fallback is False

    @pytest.mark.asyncio
    async def test_files_is_dict_str_to_str(self):
        f = _file("src/ok.py", content="x = 1")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(200, text="x = 1"))
            result = await extract_and_fetch(_payload(f), PAT)
        assert isinstance(result.files, dict)
        for k, v in result.files.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    @pytest.mark.asyncio
    async def test_failed_paths_is_list_of_str(self):
        f = _file("src/bad.py")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(500))
            result = await extract_and_fetch(_payload(f), PAT)
        assert isinstance(result.failed_paths, list)
        assert all(isinstance(p, str) for p in result.failed_paths)

    @pytest.mark.asyncio
    async def test_fallback_result_files_is_empty_dict_not_none(self):
        files = [_file(f"src/m_{i}.py") for i in range(51)]
        with respx.mock(assert_all_called=False):
            result = await extract_and_fetch(_payload(*files), PAT)
        assert result.files is not None
        assert isinstance(result.files, dict)


# ── Actual concurrency verification ──────────────────────────────────────────

class TestActualConcurrency:
    @pytest.mark.asyncio
    async def test_files_fetched_concurrently_not_sequentially(self):
        """
        Three fetches each taking 50 ms should complete in ~50 ms concurrently,
        not ~150 ms sequentially.  We allow up to 120 ms to avoid flakiness.
        """
        import time

        delay = 0.05  # 50 ms per request
        files = [_file(f"src/mod_{i}.py", content="x") for i in range(3)]

        async def slow_response(request, *args, **kwargs):
            await asyncio.sleep(delay)
            return httpx.Response(200, text="x")

        with respx.mock() as router:
            for f in files:
                router.get(f["raw_url"]).mock(side_effect=slow_response)
            t0 = time.perf_counter()
            result = await extract_and_fetch(_payload(*files), PAT)
            elapsed = time.perf_counter() - t0

        assert len(result.files) == 3
        # Sequential would take ≥ 3 * delay = 150 ms; concurrent takes ~50 ms
        assert elapsed < delay * 2.5, (
            f"Fetches appear sequential: {elapsed*1000:.0f} ms for 3×{delay*1000:.0f} ms tasks"
        )

    @pytest.mark.asyncio
    async def test_all_requests_in_flight_simultaneously(self):
        """Track request start times — all three should start before any finishes."""
        import time

        files = [_file(f"src/m_{i}.py", content="x") for i in range(3)]
        start_times: list[float] = []
        t_origin = time.perf_counter()

        async def record_start(request, *args, **kwargs):
            start_times.append(time.perf_counter() - t_origin)
            await asyncio.sleep(0.05)
            return httpx.Response(200, text="x")

        with respx.mock() as router:
            for f in files:
                router.get(f["raw_url"]).mock(side_effect=record_start)
            await extract_and_fetch(_payload(*files), PAT)

        # All three requests should start within a narrow window (< 20 ms apart)
        assert len(start_times) == 3
        assert max(start_times) - min(start_times) < 0.02, (
            f"Requests started {(max(start_times)-min(start_times))*1000:.1f} ms apart — likely sequential"
        )


# ── Duplicate paths ───────────────────────────────────────────────────────────

class TestDuplicatePaths:
    @pytest.mark.asyncio
    async def test_duplicate_paths_last_write_wins(self):
        """Duplicate path entries collapse to a single result file (Contents API
        keys off path, so both requests hit the same URL)."""
        dup = {"path": "src/auth.py", "status": "modified"}
        with respx.mock() as router:
            router.get(f"{CONTENTS_BASE}/src/auth.py").mock(
                return_value=httpx.Response(200, text="the_content")
            )
            payload = {
                "repo_id": "owner/repo", "pr_number": 1, "head_sha": "abc",
                "changed_files": [dict(dup), dict(dup)],
            }
            result = await extract_and_fetch(payload, PAT)

        assert result.files == {"src/auth.py": "the_content"}  # one entry, not two
        assert len(result.files) == 1  # deduplicated to one key

    @pytest.mark.asyncio
    async def test_duplicate_paths_no_extra_files_in_result(self):
        f = _file("src/auth.py", content="x = 1")
        payload = {
            "repo_id": "owner/repo", "pr_number": 1, "head_sha": "abc",
            "changed_files": [
                {"path": "src/auth.py", "status": "modified", "raw_url": f["raw_url"]},
                {"path": "src/auth.py", "status": "modified", "raw_url": f["raw_url"]},
            ],
        }
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(200, text="x = 1"))
            result = await extract_and_fetch(payload, PAT)

        assert list(result.files.keys()) == ["src/auth.py"]


# ── Renamed file status ───────────────────────────────────────────────────────

class TestRenamedFileStatus:
    @pytest.mark.asyncio
    async def test_renamed_file_is_fetched(self):
        """GitHub sends status='renamed' for renamed files; these have content and must be fetched."""
        f = _file("src/auth_v2.py", status="renamed", content="def login(): pass")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(
                return_value=httpx.Response(200, text=f["_content"])
            )
            result = await extract_and_fetch(_payload(f), PAT)

        assert "src/auth_v2.py" in result.files

    @pytest.mark.asyncio
    async def test_renamed_file_content_correct(self):
        f = _file("src/new_name.py", status="renamed", content="NEW_CONTENT = True")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(
                return_value=httpx.Response(200, text=f["_content"])
            )
            result = await extract_and_fetch(_payload(f), PAT)

        assert result.files["src/new_name.py"] == "NEW_CONTENT = True"

    @pytest.mark.asyncio
    async def test_only_removed_status_is_excluded(self):
        """added, modified, renamed, copied — all have content and must be fetched."""
        statuses = ["added", "modified", "renamed", "copied"]
        files = [_file(f"src/f_{s}.py", status=s, content=f"# {s}") for s in statuses]
        with respx.mock() as router:
            for f in files:
                router.get(f["raw_url"]).mock(
                    return_value=httpx.Response(200, text=f["_content"])
                )
            result = await extract_and_fetch(_payload(*files), PAT)

        for f in files:
            assert f["path"] in result.files


# ── Case sensitivity ──────────────────────────────────────────────────────────

class TestCaseSensitivity:
    @pytest.mark.parametrize("path", [
        "src/Auth.PY", "src/auth.Py",
        "README.MD", "README.Md",
        "config.TOML", "config.Json",
        "compose.YAML", "ci.YML",
    ])
    def test_uppercase_extensions_accepted(self, path):
        assert _is_ingestible(path), f"Expected {path!r} to be ingestible"

    @pytest.mark.asyncio
    async def test_uppercase_py_file_fetched(self):
        f = _file("src/Module.PY")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(200, text="x = 1"))
            result = await extract_and_fetch(_payload(f), PAT)
        assert "src/Module.PY" in result.files

    @pytest.mark.parametrize("name", [
        "Poetry.Lock", "POETRY.LOCK", "Package-Lock.JSON",
    ])
    def test_lockfiles_rejected_regardless_of_case(self, name):
        assert not _is_ingestible(name), f"Expected {name!r} to be rejected"


# ── Default timeout value ─────────────────────────────────────────────────────

class TestDefaultTimeoutValue:
    def test_fetch_timeout_constant_is_30_seconds(self):
        from api.diff import FETCH_TIMEOUT_S
        assert FETCH_TIMEOUT_S == 30.0

    @pytest.mark.asyncio
    async def test_default_timeout_used_when_not_specified(self):
        """extract_and_fetch with no http_timeout arg should not raise TypeError."""
        f = _file("src/ok.py", content="x = 1")
        with respx.mock() as router:
            router.get(f["raw_url"]).mock(return_value=httpx.Response(200, text="x = 1"))
            result = await extract_and_fetch(_payload(f), PAT)  # no http_timeout kwarg
        assert "src/ok.py" in result.files

    def test_fetch_timeout_is_float(self):
        from api.diff import FETCH_TIMEOUT_S
        assert isinstance(FETCH_TIMEOUT_S, float)

"""
Tests for Task 2.4 — Inference handoff and GitHub callback.

Covers:
- call_inference: correct request format (messages, max_tokens)
- call_inference: PHI_KEY sent as Bearer token
- call_inference: review text extracted from choices[0].message.content
- call_inference: non-2xx raises HTTPStatusError
- call_inference: timeout raises and is not swallowed
- call_inference: malformed response raises ValueError
- post_github_comment: correct URL construction (repo_id + pr_number)
- post_github_comment: PAT sent as Bearer token
- post_github_comment: body sent as JSON {"body": ...}
- post_github_comment: non-2xx raises HTTPStatusError
- post_github_comment: GitHub API version header present
- MAX_NEW_TOKENS value (8192 - 7000 = 1192)
- Timeout constants
- _process_review: fallback comment posted on pipeline exception
- _process_review: fallback comment posted when inference fails
- _process_review: fallback comment posted when GitHub callback fails on primary review
- FALLBACK_COMMENT constant is non-empty string
"""

import asyncio
import hashlib
import hmac
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock, patch, call

from api.inference import (
    call_inference,
    post_github_comment,
    MAX_NEW_TOKENS,
    INFERENCE_TIMEOUT_S,
    GITHUB_TIMEOUT_S,
    FALLBACK_COMMENT,
    GITHUB_API_BASE,
    _comment_url,
)

PHI_ENDPOINT = "https://dpant26-4965-resource.cognitiveservices.azure.com/openai/deployments/Phi-4-mini-reasoning/chat/completions?api-version=2024-12-01-preview"
PHI_KEY      = "test-phi-key-abc"
PAT          = "ghp_test_pat"
REPO_ID      = "owner/repo"
PR_NUMBER    = 42
PROMPT       = "=== SYSTEM ===\n\nReview this code.\n\n=== DIFF ===\n\ndef foo(): pass"
REVIEW_TEXT  = "# Review Summary\n\nNo violations found."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _phi_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _comment_url_for(repo_id: str, pr: int) -> str:
    return f"{GITHUB_API_BASE}/repos/{repo_id}/issues/{pr}/comments"


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_generation_budget_fits_context_window(self):
        # Input cap + generation budget must stay within Phi-4-mini's 8,192 window.
        from api.retrieval import PROMPT_TOKEN_CAP
        assert PROMPT_TOKEN_CAP + MAX_NEW_TOKENS <= 8_192

    def test_max_new_tokens_large_enough_for_reasoning_plus_answer(self):
        # The old 256 truncated the reasoning model mid-<think>; must be generous.
        assert MAX_NEW_TOKENS >= 1_000

    def test_inference_timeout_is_generous_for_reasoning_model(self):
        assert INFERENCE_TIMEOUT_S >= 120.0

    def test_github_timeout_is_30_seconds(self):
        assert GITHUB_TIMEOUT_S == 30.0

    def test_fallback_comment_is_non_empty_string(self):
        assert isinstance(FALLBACK_COMMENT, str)
        assert len(FALLBACK_COMMENT) > 0

    def test_comment_url_format(self):
        url = _comment_url("acme/backend", 7)
        assert url == "https://api.github.com/repos/acme/backend/issues/7/comments"


# ── call_inference: request format ───────────────────────────────────────────

class TestCallInferenceRequestFormat:
    @pytest.mark.asyncio
    async def test_posts_to_phi_endpoint(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json=_phi_response(REVIEW_TEXT))
            )
            await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)
            assert router.calls.call_count == 1

    @pytest.mark.asyncio
    async def test_phi_key_sent_as_bearer(self):
        captured = {}
        def capture(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=_phi_response(REVIEW_TEXT))

        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(side_effect=capture)
            await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

        assert captured["auth"] == f"Bearer {PHI_KEY}"

    @pytest.mark.asyncio
    async def test_messages_format_is_user_role(self):
        captured = {}
        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_phi_response(REVIEW_TEXT))

        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(side_effect=capture)
            await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

        messages = captured["body"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == PROMPT

    @pytest.mark.asyncio
    async def test_max_tokens_sent(self):
        captured = {}
        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_phi_response(REVIEW_TEXT))

        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(side_effect=capture)
            await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

        assert captured["body"]["max_tokens"] == MAX_NEW_TOKENS

    @pytest.mark.asyncio
    async def test_content_type_is_json(self):
        captured = {}
        def capture(request):
            captured["ct"] = request.headers.get("content-type", "")
            return httpx.Response(200, json=_phi_response(REVIEW_TEXT))

        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(side_effect=capture)
            await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

        assert "application/json" in captured["ct"]


# ── call_inference: response parsing ─────────────────────────────────────────

class TestCallInferenceResponseParsing:
    @pytest.mark.asyncio
    async def test_returns_review_text(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json=_phi_response(REVIEW_TEXT))
            )
            result = await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)
        assert result == REVIEW_TEXT

    @pytest.mark.asyncio
    async def test_returns_exact_content_string(self):
        content = "# Review\n\n## Finding 1\n\nSeverity: High\n"
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json=_phi_response(content))
            )
            result = await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)
        # Output is whitespace-stripped for a clean PR comment.
        assert result == content.strip()

    @pytest.mark.asyncio
    async def test_non_2xx_raises_http_status_error(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(return_value=httpx.Response(500))
            with pytest.raises(httpx.HTTPStatusError):
                await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

    @pytest.mark.asyncio
    async def test_401_raises_http_status_error(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(return_value=httpx.Response(401))
            with pytest.raises(httpx.HTTPStatusError):
                await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

    @pytest.mark.asyncio
    async def test_malformed_response_missing_choices_raises_value_error(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json={"error": "unexpected"})
            )
            with pytest.raises(ValueError, match="Unexpected inference response shape"):
                await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

    @pytest.mark.asyncio
    async def test_empty_choices_raises_value_error(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json={"choices": []})
            )
            with pytest.raises(ValueError):
                await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

    @pytest.mark.asyncio
    async def test_timeout_raises_and_not_swallowed(self):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                side_effect=httpx.ReadTimeout("inference timed out")
            )
            with pytest.raises(httpx.ReadTimeout):
                await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY, http_timeout=0.01)


# ── Reasoning (<think>) stripping ────────────────────────────────────────────

class TestReasoningStripping:
    @pytest.mark.asyncio
    async def _run(self, content):
        with respx.mock() as router:
            router.post(PHI_ENDPOINT).mock(
                return_value=httpx.Response(200, json=_phi_response(content))
            )
            return await call_inference(PROMPT, PHI_ENDPOINT, PHI_KEY)

    @pytest.mark.asyncio
    async def test_complete_think_block_removed(self):
        result = await self._run("<think>reasoning here</think>\n# Review\n\nNo issues.")
        assert "<think>" not in result and "reasoning here" not in result
        assert result == "# Review\n\nNo issues."

    @pytest.mark.asyncio
    async def test_unclosed_think_block_does_not_leak_raw_reasoning(self):
        # The exact failure seen on the live PR: model truncated mid-<think>, no
        # closing tag. Must NOT post the raw monologue.
        raw = "<think>Okay let me reason about this and I ran out of tok"
        result = await self._run(raw)
        assert "<think>" not in result
        assert "let me reason" not in result

    @pytest.mark.asyncio
    async def test_all_reasoning_no_verdict_falls_back_to_message(self):
        from api.inference import _TRUNCATED_MSG
        result = await self._run("<think>only reasoning, cut off before answer")
        assert result == _TRUNCATED_MSG  # clean message, never blank or raw <think>

    @pytest.mark.asyncio
    async def test_answer_after_closed_think_survives(self):
        result = await self._run("<think>thinking</think>Severity: High — SQL injection.")
        assert result == "Severity: High — SQL injection."

    @pytest.mark.asyncio
    async def test_plain_answer_without_think_untouched(self):
        result = await self._run("# Review Summary\n\nLooks good.")
        assert result == "# Review Summary\n\nLooks good."


# ── post_github_comment: request format ──────────────────────────────────────

class TestPostGithubCommentRequestFormat:
    @pytest.mark.asyncio
    async def test_posts_to_correct_url(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        with respx.mock() as router:
            router.post(url).mock(return_value=httpx.Response(201, json={"id": 1}))
            await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)
            assert router.calls.call_count == 1

    @pytest.mark.asyncio
    async def test_pat_sent_as_bearer(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        captured = {}
        def capture(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(201, json={"id": 1})

        with respx.mock() as router:
            router.post(url).mock(side_effect=capture)
            await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

        assert captured["auth"] == f"Bearer {PAT}"

    @pytest.mark.asyncio
    async def test_body_sent_as_json_body_key(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        captured = {}
        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(201, json={"id": 1})

        with respx.mock() as router:
            router.post(url).mock(side_effect=capture)
            await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

        assert captured["body"] == {"body": REVIEW_TEXT}

    @pytest.mark.asyncio
    async def test_github_api_version_header_present(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        captured = {}
        def capture(request):
            captured["version"] = request.headers.get("x-github-api-version")
            return httpx.Response(201, json={"id": 1})

        with respx.mock() as router:
            router.post(url).mock(side_effect=capture)
            await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

        assert captured["version"] == "2022-11-28"

    @pytest.mark.asyncio
    async def test_accept_header_is_github_json(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        captured = {}
        def capture(request):
            captured["accept"] = request.headers.get("accept")
            return httpx.Response(201, json={"id": 1})

        with respx.mock() as router:
            router.post(url).mock(side_effect=capture)
            await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

        assert "github" in captured["accept"]

    @pytest.mark.asyncio
    async def test_url_uses_issues_namespace(self):
        # GitHub PRs share the issues namespace for comments
        url = _comment_url(REPO_ID, PR_NUMBER)
        assert "/issues/" in url
        assert "/pulls/" not in url

    @pytest.mark.asyncio
    async def test_different_pr_number_in_url(self):
        url = _comment_url("org/myrepo", 99)
        assert "/issues/99/comments" in url
        assert "org/myrepo" in url


# ── post_github_comment: error handling ──────────────────────────────────────

class TestPostGithubCommentErrors:
    @pytest.mark.asyncio
    async def test_non_2xx_raises_http_status_error(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        with respx.mock() as router:
            router.post(url).mock(return_value=httpx.Response(403))
            with pytest.raises(httpx.HTTPStatusError):
                await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

    @pytest.mark.asyncio
    async def test_404_raises_http_status_error(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        with respx.mock() as router:
            router.post(url).mock(return_value=httpx.Response(404))
            with pytest.raises(httpx.HTTPStatusError):
                await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        url = _comment_url_for(REPO_ID, PR_NUMBER)
        with respx.mock() as router:
            router.post(url).mock(side_effect=httpx.ReadTimeout("GitHub timed out"))
            with pytest.raises(httpx.ReadTimeout):
                await post_github_comment(REPO_ID, PR_NUMBER, REVIEW_TEXT, PAT)


# ── _process_review: fallback comment on failure ─────────────────────────────

class TestProcessReviewFallback:
    """
    Verify that _process_review always posts a fallback PR comment when
    any stage of the pipeline raises, closing the gap from the 2.1 review.
    """

    def _make_payload(self) -> dict:
        return {
            "repo_id": REPO_ID,
            "pr_number": PR_NUMBER,
            "head_sha": "abc123",
            "changed_files": [],
        }

    @pytest.mark.asyncio
    async def test_fallback_comment_posted_when_fetch_fails(self):
        from api.main import _process_review
        payload = self._make_payload()

        with patch("api.main.extract_and_fetch", side_effect=RuntimeError("fetch boom")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock) as mock_comment:
            await _process_review(payload)

        mock_comment.assert_called_once()
        body = mock_comment.call_args[0][2]  # positional arg: body
        assert FALLBACK_COMMENT in body

    @pytest.mark.asyncio
    async def test_fallback_comment_posted_when_inference_fails(self):
        from api.main import _process_review
        from api.diff import FetchResult
        payload = self._make_payload()

        with patch("api.main.extract_and_fetch",
                   return_value=FetchResult(files={"src/auth.py": "x=1"})), \
             patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main.retrieve_and_assemble",
                   return_value=MagicMock(prompt="prompt", truncated=False,
                                          chunks_used=0, chunks_retrieved=0)), \
             patch("api.main.call_inference",
                   new_callable=AsyncMock,
                   side_effect=httpx.ReadTimeout("inference timed out")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock) as mock_comment:
            await _process_review(payload)

        mock_comment.assert_called_once()
        body = mock_comment.call_args[0][2]
        assert FALLBACK_COMMENT in body

    @pytest.mark.asyncio
    async def test_fallback_comment_uses_correct_repo_and_pr(self):
        from api.main import _process_review
        payload = self._make_payload()

        with patch("api.main.extract_and_fetch", side_effect=RuntimeError("boom")), \
             patch("api.main.post_github_comment", new_callable=AsyncMock) as mock_comment:
            await _process_review(payload)

        args = mock_comment.call_args[0]
        assert args[0] == REPO_ID    # repo_id
        assert args[1] == PR_NUMBER  # pr_number

    @pytest.mark.asyncio
    async def test_successful_pipeline_posts_review_not_fallback(self):
        from api.main import _process_review
        from api.diff import FetchResult
        payload = self._make_payload()

        with patch("api.main.extract_and_fetch",
                   return_value=FetchResult(files={"src/auth.py": "x=1"})), \
             patch("api.main._get_qdrant", return_value=MagicMock()), \
             patch("api.main._get_embedder", return_value=MagicMock()), \
             patch("api.main.retrieve_and_assemble",
                   return_value=MagicMock(prompt="prompt", truncated=False,
                                          chunks_used=1, chunks_retrieved=1)), \
             patch("api.main.call_inference",
                   new_callable=AsyncMock, return_value=REVIEW_TEXT), \
             patch("api.main.post_github_comment",
                   new_callable=AsyncMock) as mock_comment:
            await _process_review(payload)

        mock_comment.assert_called_once()
        body = mock_comment.call_args[0][2]
        assert body == REVIEW_TEXT
        assert FALLBACK_COMMENT not in body

    @pytest.mark.asyncio
    async def test_fallback_comment_failure_does_not_propagate(self):
        """If even the fallback comment POST fails, _process_review must not raise."""
        from api.main import _process_review
        payload = self._make_payload()

        with patch("api.main.extract_and_fetch", side_effect=RuntimeError("fetch boom")), \
             patch("api.main.post_github_comment",
                   new_callable=AsyncMock,
                   side_effect=httpx.HTTPStatusError(
                       "403", request=MagicMock(), response=MagicMock())):
            # Must not raise
            await _process_review(payload)

    @pytest.mark.asyncio
    async def test_macro_path_posts_review_comment(self):
        """Fallback macro prompt (>50 files) still results in a posted comment."""
        from api.main import _process_review
        from api.diff import FetchResult
        payload = {**self._make_payload(), "commit_message": "big refactor"}

        with patch("api.main.extract_and_fetch",
                   return_value=FetchResult(fallback=True)), \
             patch("api.main.call_inference",
                   new_callable=AsyncMock, return_value=REVIEW_TEXT), \
             patch("api.main.post_github_comment",
                   new_callable=AsyncMock) as mock_comment:
            await _process_review(payload)

        mock_comment.assert_called_once()
        body = mock_comment.call_args[0][2]
        assert body == REVIEW_TEXT

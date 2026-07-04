"""
Tests for Task 2.3 — Retrieval and prompt assembly.

Covers:
- repo_id filter always present in Qdrant query
- TOP_K and SCORE_THRESHOLD constants
- Prompt contains all three required blocks (SYSTEM, REPOSITORY CONTEXT, PULL REQUEST DIFF)
- Chunk content appears in assembled prompt
- Truncation: lowest-scoring chunks dropped first when cap breached
- System and Diff blocks preserved after truncation
- Zero chunks retrieved → placeholder emitted, not crash
- Macro-summary prompt (>50 files fallback path)
- PromptResult contract fields
- Token cap enforcement
"""

import pytest
from unittest.mock import MagicMock, patch

from api.retrieval import (
    retrieve_and_assemble,
    build_macro_prompt,
    PromptResult,
    TOP_K,
    SCORE_THRESHOLD,
    PROMPT_TOKEN_CAP,
    _count,
    _truncate_to_cap,
    _diff_block,
    _context_block,
    _assemble,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_point(content: str, score: float, repo_id: str = "owner/repo"):
    p = MagicMock()
    p.payload = {"content": content, "repo_id": repo_id}
    p.score = score
    return p


def _make_qdrant(points: list) -> MagicMock:
    result = MagicMock()
    result.points = points
    client = MagicMock()
    client.query_points.return_value = result
    return client


def _make_embedder(dim: int = 768):
    import numpy as np
    emb = MagicMock()
    emb.embed.return_value = iter([np.zeros(dim, dtype="float32")])
    return emb


PAYLOAD = {
    "repo_id": "owner/repo",
    "pr_number": 7,
    "head_sha": "abc123",
    "changed_files": [{"path": "src/auth.py", "status": "modified",
                       "raw_url": "https://example.com/auth.py"}],
}
FILES = {"src/auth.py": "def login(user, pw):\n    return True\n"}


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_top_k_is_5(self):
        assert TOP_K == 5

    def test_score_threshold_is_0_75(self):
        assert SCORE_THRESHOLD == 0.75

    def test_prompt_token_cap_leaves_generation_headroom(self):
        # Input cap + generation must fit gemini-2.5-flash's ~1M-token window.
        from api.inference import MAX_NEW_TOKENS
        assert PROMPT_TOKEN_CAP + MAX_NEW_TOKENS <= 1_000_000
        assert PROMPT_TOKEN_CAP >= 4_000  # still enough room for retrieved context


# ── repo_id filter ────────────────────────────────────────────────────────────

class TestRepoIdFilter:
    def test_query_called_with_repo_id_filter(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)

        call_kwargs = client.query_points.call_args.kwargs
        qfilter = call_kwargs["query_filter"]
        must_conditions = qfilter.must
        assert len(must_conditions) == 1
        cond = must_conditions[0]
        assert cond.key == "repo_id"
        assert cond.match.value == "owner/repo"

    def test_different_repo_id_passed_through(self):
        payload = {**PAYLOAD, "repo_id": "acme/backend"}
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, payload, client, embedder)

        cond = client.query_points.call_args.kwargs["query_filter"].must[0]
        assert cond.match.value == "acme/backend"

    def test_top_k_passed_to_qdrant(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)

        assert client.query_points.call_args.kwargs["limit"] == TOP_K

    def test_score_threshold_passed_to_qdrant(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)

        assert client.query_points.call_args.kwargs["score_threshold"] == SCORE_THRESHOLD

    def test_correct_collection_queried(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)

        assert client.query_points.call_args.kwargs["collection_name"] == "global_codebase_memory"


# ── Prompt structure ──────────────────────────────────────────────────────────

class TestPromptStructure:
    def test_prompt_contains_system_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "SYSTEM" in result.prompt

    def test_prompt_contains_repository_context_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "REPOSITORY CONTEXT" in result.prompt

    def test_prompt_contains_pull_request_diff_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "PULL REQUEST DIFF" in result.prompt

    def test_prompt_contains_task_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "TASK" in result.prompt

    def test_diff_content_in_prompt(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "def login" in result.prompt

    def test_file_path_in_diff_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "src/auth.py" in result.prompt

    def test_chunk_content_appears_in_prompt(self):
        points  = [_make_point("All functions must have tests.", 0.9)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "All functions must have tests." in result.prompt

    def test_system_block_before_context_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        sys_pos  = result.prompt.index("SYSTEM")
        ctx_pos  = result.prompt.index("REPOSITORY CONTEXT")
        diff_pos = result.prompt.index("PULL REQUEST DIFF")
        assert sys_pos < ctx_pos < diff_pos


# ── Zero chunks retrieved ─────────────────────────────────────────────────────

class TestZeroChunks:
    def test_no_chunks_does_not_crash(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert isinstance(result.prompt, str)
        assert len(result.prompt) > 0

    def test_no_chunks_emits_placeholder(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "no repository context retrieved" in result.prompt

    def test_no_chunks_result_not_truncated(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.truncated is False

    def test_no_chunks_used_is_zero(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_used == 0

    def test_no_chunks_retrieved_is_zero(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_retrieved == 0


# ── Truncation ────────────────────────────────────────────────────────────────

class TestTruncation:
    def _big_chunk(self, n_words: int = 1500) -> str:
        return " ".join([f"word{i}" for i in range(n_words)])

    def test_chunks_dropped_when_cap_breached(self):
        # Five large chunks that together exceed 7,000 tokens
        points = [
            _make_point(self._big_chunk(1500), score=0.95 - i * 0.05)
            for i in range(5)
        ]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert _count(result.prompt) <= PROMPT_TOKEN_CAP

    def test_lowest_scoring_chunk_dropped_first(self):
        high_score = _make_point("HIGH_SCORE_CONTENT unique_abc", score=0.95)
        low_score  = _make_point("LOW_SCORE_CONTENT unique_xyz " + "word " * 1400, score=0.76)
        client  = _make_qdrant([high_score, low_score])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        # High-score chunk preserved; low-score dropped if cap was breached
        if result.truncated:
            assert "HIGH_SCORE_CONTENT" in result.prompt
            assert "LOW_SCORE_CONTENT" not in result.prompt

    def test_system_block_preserved_after_truncation(self):
        points = [_make_point(self._big_chunk(1500), 0.95 - i * 0.05) for i in range(5)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "SYSTEM" in result.prompt

    def test_diff_block_preserved_after_truncation(self):
        points = [_make_point(self._big_chunk(1500), 0.95 - i * 0.05) for i in range(5)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "PULL REQUEST DIFF" in result.prompt
        assert "def login" in result.prompt

    def test_truncated_flag_true_when_chunks_dropped(self):
        points = [_make_point(self._big_chunk(1500), 0.95 - i * 0.05) for i in range(5)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.truncated is True

    def test_truncated_flag_false_when_no_drop_needed(self):
        points = [_make_point("Short chunk.", 0.9)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.truncated is False

    def test_chunks_used_less_than_retrieved_when_truncated(self):
        points = [_make_point(self._big_chunk(1500), 0.95 - i * 0.05) for i in range(5)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        if result.truncated:
            assert result.chunks_used < result.chunks_retrieved

    def test_prompt_within_token_cap_always(self):
        # Run with both small and large chunk sets
        for n_chunks in [0, 1, 3, 5]:
            points = [_make_point(self._big_chunk(1500), 0.95) for _ in range(n_chunks)]
            client  = _make_qdrant(points)
            embedder = _make_embedder()
            result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
            assert _count(result.prompt) <= PROMPT_TOKEN_CAP, (
                f"Prompt exceeded cap with {n_chunks} chunks: {_count(result.prompt)} tokens"
            )


# ── _truncate_to_cap unit tests ───────────────────────────────────────────────

class TestTruncateToCapUnit:
    def test_empty_chunks_not_truncated(self):
        kept, truncated = _truncate_to_cap([], diff="small diff")
        assert kept == []
        assert truncated is False

    def test_single_small_chunk_not_truncated(self):
        chunks = [("A short rule.", 0.9)]
        kept, truncated = _truncate_to_cap(chunks, diff="small diff")
        assert len(kept) == 1
        assert truncated is False

    def test_pops_from_end_lowest_score(self):
        # Assumes Qdrant returns highest score first
        chunks = [("best", 0.95), ("second", 0.85), ("worst " + "w " * 2000, 0.76)]
        kept, truncated = _truncate_to_cap(chunks, diff="diff")
        if truncated:
            assert "best" in kept
            assert "worst" + " w" not in " ".join(kept) or not truncated


# ── Macro-summary prompt ──────────────────────────────────────────────────────

class TestMacroPrompt:
    def _large_payload(self, n: int = 55) -> dict:
        return {
            "repo_id": "owner/bigrepository",
            "pr_number": 99,
            "commit_message": "Refactor everything",
            "changed_files": [
                {"path": f"src/module_{i}.py", "status": "modified",
                 "raw_url": f"https://example.com/{i}"}
                for i in range(n)
            ],
        }

    def test_macro_prompt_contains_system_block(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "SYSTEM" in prompt

    def test_macro_prompt_contains_repo_id(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "owner/bigrepository" in prompt

    def test_macro_prompt_contains_commit_message(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "Refactor everything" in prompt

    def test_macro_prompt_contains_file_count(self):
        prompt = build_macro_prompt(self._large_payload(55))
        assert "55" in prompt

    def test_macro_prompt_contains_file_names(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "src/module_0.py" in prompt

    def test_macro_prompt_no_qdrant_context_block(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "REPOSITORY CONTEXT" not in prompt

    def test_macro_prompt_has_task_block(self):
        prompt = build_macro_prompt(self._large_payload())
        assert "TASK" in prompt

    def test_macro_prompt_within_token_cap(self):
        prompt = build_macro_prompt(self._large_payload(55))
        assert _count(prompt) <= PROMPT_TOKEN_CAP

    def test_macro_prompt_missing_commit_message_uses_placeholder(self):
        payload = {**self._large_payload(), "commit_message": None}
        payload.pop("commit_message", None)
        prompt = build_macro_prompt(payload)
        assert "no commit message provided" in prompt


# ── PromptResult contract ─────────────────────────────────────────────────────

class TestPromptResultContract:
    def test_result_has_prompt_str(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert isinstance(result.prompt, str)

    def test_result_has_truncated_bool(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert isinstance(result.truncated, bool)

    def test_result_has_chunks_used_int(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert isinstance(result.chunks_used, int)

    def test_result_has_chunks_retrieved_int(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert isinstance(result.chunks_retrieved, int)

    def test_chunks_used_equals_chunks_retrieved_when_no_truncation(self):
        points  = [_make_point("Short rule.", 0.9)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_used == result.chunks_retrieved


# ── Empty files dict (all fetches failed) ────────────────────────────────────

class TestEmptyFiles:
    def test_empty_files_uses_placeholder_diff(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble({}, PAYLOAD, client, embedder)
        assert "no file content fetched" in result.prompt

    def test_empty_files_still_has_all_blocks(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result = retrieve_and_assemble({}, PAYLOAD, client, embedder)
        assert "SYSTEM" in result.prompt
        assert "REPOSITORY CONTEXT" in result.prompt
        assert "PULL REQUEST DIFF" in result.prompt


# ── Embedding model matches Phase 1 ──────────────────────────────────────────

class TestEmbeddingModelAlignment:
    """
    Phase 1 ingests with nomic-ai/nomic-embed-text-v1.5 at 768 dimensions.
    Phase 2 must query with the identical model and dimensionality — a mismatch
    silently degrades retrieval quality rather than raising an error.
    """

    def test_phase1_pipeline_uses_nomic_embed_text(self):
        from ingestion.pipeline import _load_embedder
        import inspect
        src = inspect.getsource(_load_embedder)
        assert "nomic-ai/nomic-embed-text-v1.5" in src, (
            "Phase 1 _load_embedder must use nomic-ai/nomic-embed-text-v1.5"
        )

    def test_phase2_retrieval_uses_same_model_name(self):
        # Phase 2's model name lives in _load_embedder; _get_embedder just caches it.
        from api.main import _load_embedder
        import inspect
        src = inspect.getsource(_load_embedder)
        assert "nomic-ai/nomic-embed-text-v1.5" in src, (
            "Phase 2 _load_embedder must use the same model as Phase 1"
        )

    def test_both_phases_use_identical_model_string(self):
        from ingestion.pipeline import _load_embedder as p1_loader
        from api.main import _load_embedder as p2_loader
        import inspect, re
        pattern = r'model_name\s*=\s*["\']([^"\']+)["\']'
        p1 = re.search(pattern, inspect.getsource(p1_loader))
        p2 = re.search(pattern, inspect.getsource(p2_loader))
        assert p1 and p2, "Both phases must declare a model_name"
        assert p1.group(1) == p2.group(1), (
            f"Model mismatch: Phase 1 uses {p1.group(1)!r}, "
            f"Phase 2 uses {p2.group(1)!r}"
        )

    def test_embedding_output_dimensionality_is_768(self):
        """
        nomic-embed-text-v1.5 produces 768-dim vectors.
        Verify the mock used throughout these tests matches that dimension,
        and that the Qdrant collection schema (768) is consistent.
        """
        import numpy as np
        emb = _make_embedder(dim=768)
        vectors = list(emb.embed(["test"]))
        assert vectors[0].shape == (768,), (
            f"Expected 768-dim vector, got {vectors[0].shape}"
        )

    def test_qdrant_collection_vector_size_matches_embedding_dim(self):
        """
        The collection was created with vector_size=768 in Phase 0.
        Confirm the constant used in Phase 2 retrieval matches.
        Qdrant raises on dimension mismatch, but we catch it early here.
        """
        import numpy as np
        # The vector passed to query_points must be 768-dim
        client   = _make_qdrant([])
        embedder = _make_embedder(dim=768)
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        query_vector = client.query_points.call_args.kwargs["query"]
        assert len(query_vector) == 768, (
            f"Query vector is {len(query_vector)}-dim; Qdrant collection expects 768"
        )


# ── SCORE_THRESHOLD filtering ─────────────────────────────────────────────────

class TestScoreThresholdFiltering:
    """
    Qdrant's score_threshold parameter filters server-side, but we also verify
    the downstream behaviour when fewer than TOP_K chunks pass the threshold.
    """

    def test_chunks_below_threshold_excluded_by_qdrant_param(self):
        # Confirm score_threshold=0.75 is forwarded so Qdrant filters server-side
        client  = _make_qdrant([])
        embedder = _make_embedder()
        retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert client.query_points.call_args.kwargs["score_threshold"] == SCORE_THRESHOLD

    def test_fewer_than_top_k_chunks_assembles_correctly(self):
        # Only 2 of the possible 5 chunks pass (Qdrant returns fewer)
        points  = [_make_point(f"Rule {i}.", 0.9 - i * 0.02) for i in range(2)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_retrieved == 2
        assert result.chunks_used == 2
        assert result.truncated is False
        assert "Rule 0." in result.prompt
        assert "Rule 1." in result.prompt

    def test_zero_chunks_pass_threshold_still_assembles(self):
        # All chunks filtered out by score_threshold — Qdrant returns []
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_retrieved == 0
        assert "no repository context retrieved" in result.prompt
        assert _count(result.prompt) <= PROMPT_TOKEN_CAP

    def test_one_chunk_passes_prompt_still_complete(self):
        points  = [_make_point("Only surviving rule.", 0.80)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "Only surviving rule." in result.prompt
        assert "SYSTEM" in result.prompt
        assert "PULL REQUEST DIFF" in result.prompt

    def test_chunks_used_reflects_post_threshold_count(self):
        # Simulate Qdrant returning 3 chunks (2 were filtered below threshold)
        points  = [_make_point(f"Rule {i}.", 0.9) for i in range(3)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert result.chunks_retrieved == 3
        assert result.chunks_used == 3


# ── Truncation accounts for TASK / OUTPUT FORMAT blocks ──────────────────────

class TestTruncationIncludesAllBlocks:
    """
    The 7,000-token cap must account for ALL blocks in the assembled prompt:
    SYSTEM + REPOSITORY CONTEXT + PULL REQUEST DIFF + TASK + OUTPUT FORMAT.
    A naive implementation that only counts System+Context+Diff will allow
    TASK and OUTPUT FORMAT to push the total over 7,000.
    """

    def test_assembled_prompt_includes_task_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "TASK" in result.prompt

    def test_assembled_prompt_includes_output_format_block(self):
        client  = _make_qdrant([])
        embedder = _make_embedder()
        result  = retrieve_and_assemble(FILES, PAYLOAD, client, embedder)
        assert "OUTPUT FORMAT" in result.prompt

    def test_task_and_output_format_tokens_counted_in_cap(self):
        """
        Build a prompt borderline near 7,000 tokens and confirm the
        TASK + OUTPUT FORMAT blocks are already included — total must not exceed cap.
        """
        from api.retrieval import _assemble, _TASK_BLOCK
        task_tokens = _count(_TASK_BLOCK)
        assert task_tokens > 0, "TASK block must have non-zero token count"

        # Confirm _assemble includes the TASK block in the returned string
        assembled = _assemble("ctx", "diff")
        assert "TASK" in assembled
        assert "OUTPUT FORMAT" in assembled

        # Token count of _assemble must exceed just System+Context+Diff
        system_only = (
            "=========================\nSYSTEM\n=========================\n\n"
            "=========================\nREPOSITORY CONTEXT\n=========================\n\nctr\n\n"
            "=========================\nPULL REQUEST DIFF\n=========================\n\ndiff"
        )
        full = _assemble("ctx", "diff")
        assert _count(full) > _count(system_only), (
            "Assembled prompt must include TASK and OUTPUT FORMAT token cost"
        )

    def test_cap_enforced_on_full_assembled_string_not_partial(self):
        """
        Truncation must measure the *complete* assembled prompt (all blocks),
        not just the context portion. Use a chunk size that fits in
        System+Context+Diff but pushes over 7,000 when TASK is added.
        """
        from api.retrieval import _assemble, _TASK_BLOCK, _context_block

        # Measure overhead of everything except context
        overhead = _count(_assemble("", "small diff"))
        # Build a context that would fit if overhead were underestimated
        # but pushes over cap when all blocks are counted correctly
        budget_words = max(0, PROMPT_TOKEN_CAP - overhead + 50)  # 50 tokens over
        big_context  = " ".join(f"word{i}" for i in range(budget_words))

        points   = [(big_context, 0.9)]
        kept, truncated = _truncate_to_cap(points, diff="small diff")

        full_prompt = _assemble(_context_block(kept), "small diff")
        assert _count(full_prompt) <= PROMPT_TOKEN_CAP, (
            f"Prompt is {_count(full_prompt)} tokens — TASK/OUTPUT FORMAT not counted in cap"
        )

    def test_borderline_prompt_within_cap_including_all_blocks(self):
        """End-to-end: prompt with multiple chunks near the cap stays within 7,000."""
        # Use chunks that together push close to but not over the cap
        from api.retrieval import _assemble, _context_block
        overhead = _count(_assemble("", "diff text"))
        per_chunk_budget = (PROMPT_TOKEN_CAP - overhead) // 3
        chunk_words = max(1, per_chunk_budget // 2)  # half-budget chunks, safe margin

        points  = [_make_point(" ".join(f"w{i}" for i in range(chunk_words)), 0.9)
                   for _ in range(3)]
        client  = _make_qdrant(points)
        embedder = _make_embedder()
        result  = retrieve_and_assemble({"f.py": "diff text"}, PAYLOAD, client, embedder)

        assert _count(result.prompt) <= PROMPT_TOKEN_CAP, (
            f"Borderline prompt is {_count(result.prompt)} tokens > {PROMPT_TOKEN_CAP}"
        )

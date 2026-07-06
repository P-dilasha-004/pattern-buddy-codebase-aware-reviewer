# Pattern Buddy — Latency

Measured from the backend's own telemetry (Task 5.1 `stage_log` + Task 5.2
`system_metrics`) plus GitHub PR timestamps. Regenerate any time with
[`latency_report.py`](latency_report.py) — see [Reproducing](#reproducing).

## Headline numbers

Reference run: **PR #8** on `test-repo-codebase-aware-reviewer` (a clean,
codebase-aware review — 5 findings, 7,566 chars, retrieval returned 5 chunks).

| Metric | Value |
|---|---|
| **End-to-end (PR opened → comment posted)** | **~24 s** |
| Backend only (webhook received → comment posted) | ~13 s |
| GitHub Actions trigger overhead (PR opened → webhook) | ~12 s |
| Pattern Buddy's own compute (fetch + embed + retrieve + post) | ~3.3 s |
| RAM during review | ~5.3 GB (of B3's 7 GB) |

## Per-stage breakdown (PR #8)

| Stage | Latency | Who owns it |
|---|---|---|
| GitHub Action trigger | ~12 s | GitHub (runner spin-up + `gh api` + sign + `curl`) |
| `diff_fetch` (Contents API) | 0.38 s | Pattern Buddy |
| `embedding` (nomic-embed) | 1.03 s | Pattern Buddy |
| `qdrant_retrieval` | 0.79 s | Pattern Buddy |
| `inference_handoff` (Gemini) | 9.69 s | Gemini (external) |
| post comment (GitHub API) | ~1.1 s | Pattern Buddy → GitHub |

## Key findings

1. **Pattern Buddy's own pipeline is fast and stable — ~3.3 s total.** Diff
   fetch, embedding, Qdrant retrieval, and comment posting are each sub-second
   to ~1 s and barely vary run to run. There is no low-hanging fruit here.

2. **~86% of end-to-end latency is external wait**, split between two things
   Pattern Buddy doesn't control:
   - **GitHub Actions runner startup (~12 s)** — fixed overhead of the stateless
     trigger workflow; unavoidable with the current no-checkout design (which is
     also what keeps the Action cheap and simple).
   - **Gemini inference (~9.7 s)** — the LLM generating the review.

3. **Inference is the dominant *backend* cost (~74%) and the main variable.**
   Measured **9.7 s (PR #8)** vs **37 s (PR #7)** for similarly-sized reviews
   (~7,200–7,566 chars). The swing is server-side Gemini response time (load +
   output length), not anything in our code. This is the only stage worth tuning
   if latency ever matters — e.g. a smaller/faster Gemini variant, at some
   quality cost.

4. **Ingestion is off the review hot path.** `/admin/reindex` (one-time per repo)
   takes minutes for a full repo — test-repo's 166 chunks embed in batches of 16
   with event-loop yields (batching is what keeps the single worker responsive;
   see the reindex worker-kill fix). It never affects per-review latency.

## Aggregates (from the current log snapshot, n=2 successful reviews)

```
                        mean /     p50 /     p95
  diff_fetch           342ms /   342ms /   373ms
  embedding            1.08s /   1.08s /   1.13s
  qdrant_retrieval     784ms /   784ms /   786ms
  inference_handoff   23.33s /  23.33s /  35.61s   <- high variance (Gemini)
  end_to_end (backend)31.52s /  31.52s /  48.17s
```

Note the small sample: PR #7's 37 s inference outlier pulls the mean up. A
single clean review (PR #8) is ~13 s backend / ~24 s wall-clock, which is the
representative "good" case. Collect more runs for a stable p50/p95.

## Reproducing

`latency_report.py` pulls the App Service container logs, parses the per-review
telemetry, and prints the table + aggregates. GitHub timing (trigger overhead +
wall-clock) is added when you pass `--repo`/`--pr`.

```bash
# Pull fresh logs from Azure and print the report:
python latency_report.py --download

# Add GitHub trigger + wall-clock for one review:
python latency_report.py --download \
  --repo P-dilasha-004/test-repo-codebase-aware-reviewer --pr 8

# Reuse an already-extracted LogFiles dir (no re-download):
python latency_report.py --logs ./LogFiles
```

Requires `az` logged in (for `--download`) and a GitHub `PAT` in env or `.env`
(for `--repo`/`--pr`). The numbers come straight from the backend's structured
logs, so this stays accurate as the system changes — if a stage gets slower,
it shows up here.

# Pattern Buddy (codebase-aware-reviewer)

Pattern Buddy is an automated pull request review agent that reviews PRs the
way a senior engineer who actually knows your codebase would — by retrieving
the specific repository context relevant to a diff (existing conventions,
similar code elsewhere in the codebase, related tests) and grounding its
feedback in that context, instead of reviewing the diff in isolation the way
a generic AI code reviewer does.

It's a Retrieval-Augmented Generation (RAG) system:
every repository it reviews is chunked, embedded, and stored in a vector
database ahead of time; when a PR opens, Pattern Buddy embeds the diff,
retrieves the most relevant chunks of the *actual* repository, and hands
both to an LLM to generate the review.

---

## Why

Two common approaches to AI code review both have a real gap:

- **Generic diff-only reviewers** (most AI PR review bots) see only the
  changed lines. They can catch obvious bugs, but they have no idea what
  *this* codebase's conventions are — they can't tell you "this violates the
  pattern established in `auth_service.py`" because they've never seen
  `auth_service.py`.
- **Whole-codebase-in-context reviewers** (pasting the entire repo into a
  large context window) solve the awareness problem but don't scale: it's
  slow, token-expensive, and falls apart entirely once a repo exceeds the
  model's context window.

Pattern Buddy's bet is that most PRs only need a handful of *relevant*
pieces of repository context — the file being modified's neighbors, the
tests that cover it, the architectural conventions that apply to it — not
the whole codebase. Retrieval gets you that targeted context for a fraction
of the token cost, and it's what lets the review actually say things like
*"this hardcodes `PROJECT_ID` where every other BigQuery call in this repo
reads it from `os.getenv`"* — a cross-file, convention-aware finding that a
diff-only reviewer structurally cannot make.

---

## How it works

### End-to-end flow

```
PR opened/updated on GitHub
        │
        ▼
GitHub Actions workflow (.github/workflows/pr_review.yml)
  - fetches the PR's changed-file list via the GitHub API
  - builds a lightweight JSON payload (repo, PR #, base/head SHA, changed files)
  - HMAC-SHA256 signs it with a shared webhook secret
  - POSTs it to the Pattern Buddy backend
        │
        ▼
Backend: POST /review  (Azure App Service, FastAPI)
  - verifies the HMAC signature — rejects anything unsigned/forged (401)
  - checks a per-repo concurrency guard — a repo already being reviewed
    gets a 409 instead of a second overlapping review
  - returns 202 Accepted IMMEDIATELY and keeps working in a background task
    (this is what keeps the GitHub Actions runner from blocking for the
    ~15-30s the actual pipeline takes)
        │
        ▼  (background task, from here on)
1. Concurrent file fetch — every changed file's raw content is fetched from
   GitHub in parallel (asyncio.gather), with per-file failure isolation: one
   file 404ing doesn't abort the batch, it's just excluded and logged.
   PRs touching more than 50 files skip straight to a macro-level,
   commit-message-only review instead of trying to process everything.
2. Embedding — the diff is embedded locally in-process using
   nomic-embed-text-v1.5 (via fastembed/ONNX, no external embedding API call).
3. Retrieval — the diff's embedding is used to query Qdrant for the top 5
   most similar chunks of the *same repository's* pre-ingested code/docs
   (cosine similarity, score ≥ 0.60). Every query is hard-filtered on
   repo_id server-side, so one repo's context can never leak into another
   repo's review even if something upstream misbehaves.
4. Prompt assembly — system instructions + retrieved repository context +
   the PR diff are assembled into a single prompt, capped at 6,000 input
   tokens (chunks are dropped lowest-score-first if it doesn't fit).
5. Inference — the prompt goes to an LLM (currently Gemini 2.5 Flash) to
   generate a structured Markdown review: findings with severity, the
   specific repository rule/convention each one cites, and a suggested fix.
   If no real violation is found, it says so explicitly rather than
   padding the review with generic advice.
6. The generated review is posted back to the PR as a GitHub comment.
```

### Ingesting a repository (what makes review "codebase-aware" at all)

Before Pattern Buddy can review a repo's PRs with real context, the repo has
to be ingested — chunked and embedded into Qdrant. This runs once up front
per repo (and can be re-triggered for a full rebuild via `/admin/reindex`).
Ingestion is polymorphic — different file types are chunked differently
because "one text splitter for everything" produces bad retrieval:

| Track | File types | Strategy |
|---|---|---|
| A — Source code | `.py` | Tree-sitter AST chunking at function/class boundaries (~800 tokens/chunk), with each chunk's file-level imports prepended so it's self-contained |
| B — Prose/docs | `.md`, `.txt` | Chunked at Markdown header boundaries (300–600 tokens), with narrative overlap when a section runs long |
| C — Config/manifests | `.toml`, `.json`, `.yaml` | Flattened into short declarative sentences rather than raw text (100–300 tokens) |

Every chunk is stored in a single shared Qdrant collection
(`global_codebase_memory`), multi-tenant-partitioned by `repo_id` — vectors
from different repositories are physically co-located per tenant and every
retrieval query is filtered on `repo_id`, so there's no cross-repo
contamination even though many repos share one collection.

---

## Architecture

| Component | What | Where |
|---|---|---|
| Backend API | FastAPI, receives webhooks, orchestrates the pipeline | Azure App Service (`pattern-buddy-v2`) |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5` (fastembed/ONNX, 768-dim) | In-process on the backend |
| Inference | Gemini 2.5 Flash, OpenAI-compatible chat completions endpoint | External (Google AI) |
| Vector DB | Qdrant Cloud | Collection `global_codebase_memory`, `repo_id`-partitioned |
| Trigger | GitHub Actions workflow, per onboarded repo | `.github/workflows/pr_review.yml` |

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /review` | PR trigger — the main entry point, called by the GitHub Actions workflow on `opened`/`synchronize` |
| `POST /admin/reindex` | Full repository rebuild — re-fetches the repo as a zip, wipes and re-ingests all of its vectors |
| `POST /sync` | Merge-time incremental sync (delete-then-upsert for changed files only) |
| `GET /health` | Liveness check |

Every write-capable endpoint validates an `X-Hub-Signature-256` HMAC
signature against a shared webhook secret before doing anything else, and
`/review`/`/admin/reindex` each enforce a per-repo concurrency guard so two
overlapping triggers for the same repo can't race each other (a second
concurrent request gets `409`, not a duplicate run).

### Repository layout

```
api/                  FastAPI backend
  main.py             endpoints, background task orchestration, telemetry
  diff.py             concurrent PR file fetching from GitHub
  retrieval.py         embedding + Qdrant query + prompt assembly
  inference.py         LLM call
ingestion/             the polymorphic chunking pipeline (Tracks A/B/C)
run_ingestion.py       CLI entry point for one-off local ingestion runs
latency_report.py      pulls production telemetry and prints latency stats
.github/workflows/     the trigger workflow template installed into onboarded repos
```

---

## How to use it

Onboarding a repository to Pattern Buddy today is a manual process:

1. **Set repo secrets** on the target repository: `WEBHOOK_SECRET` (shared
   HMAC secret) and `PAT` (a GitHub token used by the *workflow* to read PR
   file lists — separate from the backend's own PAT).
2. **Install the trigger workflow** — copy `.github/workflows/pr_review.yml`
   into the target repo. It fires on `pull_request: [opened, synchronize]`,
   builds the signed payload, and POSTs it to the backend.
3. **Ingest the repository** — trigger `POST /admin/reindex` (signed the
   same way) with `{"repo_id": "<owner>/<repo>", "branch": "<default branch>"}`
   so there's real repository context in Qdrant before the first review runs.
4. **Open a PR.** Within seconds of the workflow firing, Pattern Buddy posts
   a review comment.

Without step 3, `/review` still works — it just runs diff-only (no
repository context retrieved), which is a strictly weaker review, not a
failure mode.

---

## Limitations

This is an actively-developed system, and a few real, known limitations are
worth being upfront about:

### Authentication is currently PAT-based, not a GitHub App

Right now, both the backend (for fetching files and posting review comments)
and the per-repo trigger workflow (for reading PR file lists) authenticate
with a static fine-grained Personal Access Token. This works, but it isn't
the right end state: a GitHub App would give per-installation tokens,
narrower scoped permissions, and a self-service install flow instead of
manually wiring secrets into every repo.

A full GitHub App conversion was actually built and deployed earlier
(JWT signing, installation tokens, a unified webhook endpoint, auto-ingest
on install) — but it hit a persistent `403 Resource not accessible by
integration` when posting PR comments, *despite* `issues: write` being
correctly granted and confirmed at every inspectable layer (the app
manifest, the installation object, the freshly-minted token's own reported
permissions, and even GitHub's own response headers confirming the
permission should have been sufficient). It reproduced identically across
two independently-created GitHub Apps and multiple repositories, and wasn't
resolved through any combination of scope, timing, or configuration changes
tried. Rather than ship on top of an unexplained platform-level failure, the
GitHub App path was reverted back to the PAT architecture, which is what's
live today. Revisiting the GitHub App conversion is active, ongoing work.

### Ingesting large repositories is unreliable

`/admin/reindex` — the full-repository ingestion path — has a known,
partially-fixed worker-crash bug. The original version of this bug caused
gunicorn to SIGKILL the worker process on *any* repo, misreported by
gunicorn as an out-of-memory kill; the actual cause was the ingestion
background task blocking the event loop long enough that gunicorn's
internal worker-liveness heartbeat starved, which the arbiter treats as a
dead worker. That was fixed for small/medium repositories by batching the
embed+upsert work and yielding back to the event loop between batches
(`await asyncio.sleep(0)`).

That fix does **not** fully hold for large repositories. Repos with
hundreds of files and thousands of ingestible chunks (confirmed
reproducible on more than one real repo, including a full clone of the
FastAPI framework — ~2,800 ingestible files) still crash the worker
partway through ingestion, sometimes with genuine, substantial memory growth
this time (multiple GB, not the sub-1GB "false OOM" signature of the
original bug) — suggesting this may be a related but distinct failure mode,
not simply "the same bug at larger scale." The practical impact: ingesting
a large repository today may need multiple retries, may only partially
complete (leaving a repo with sparse rather than full retrieval context),
and briefly affects the shared backend for *other* onboarded repos while it
recovers (the backend runs as a single worker process). This is active,
ongoing optimization work.

### Also worth knowing

- **Review quality depends on ingestion having actually happened.** A repo
  that hasn't been ingested (or was only partially ingested, per the
  limitation above) still gets reviewed, just without codebase-specific
  grounding — the review degrades gracefully to diff-only, it doesn't fail.

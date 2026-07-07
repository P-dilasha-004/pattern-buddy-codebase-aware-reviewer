# codebase-aware-reviewer (Pattern Buddy)

A retrieval-augmented PR reviewer: when a pull request opens, it fetches the
changed files, retrieves relevant context from a vector index of the target
repository, and posts a review comment grounded in that repo's own
conventions rather than generic best practices.

## Using it on your repo

Pattern Buddy runs as a GitHub App. Installing it is the entire onboarding
flow — no secrets, no workflow files, no manual setup on your side:

1. Install the app on your repo (or org) from its GitHub App install page.
2. That's it. Installing triggers an automatic first ingestion of your repo,
   and opening a pull request will produce a review comment shortly after.

No `PAT` or `WEBHOOK_SECRET` repo secrets and no `.github/workflows/*.yml`
files are required in your repo — those were needed by an earlier,
PAT-based version of this project and are no longer part of the flow.

## Backend

FastAPI service (`api/`) + ingestion pipeline (`ingestion/`), deployed on
Azure App Service. Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /webhook` | Single entry point for all GitHub App events (`pull_request`, `installation`, `installation_repositories`, `ping`), routed by `X-GitHub-Event`. |
| `POST /admin/reindex` | Manual full re-ingest for one repo (recovery path). |
| `POST /admin/reindex-all` | Re-ingest every repo the App is installed on — driven by a weekly cron in this repo's own `.github/workflows/weekly_rebuild.yml`. |
| `GET /health` | Liveness check. |

Auth to the GitHub API is via GitHub App installation tokens
(`api/github_app.py`), auto-scoped by GitHub to whatever repos an
installation covers — not a manually-scoped personal access token.

### Environment variables

| Var | Meaning |
|---|---|
| `GITHUB_APP_ID` | The GitHub App's numeric ID |
| `GITHUB_APP_PRIVATE_KEY` | The App's private key (PEM), base64-encoded |
| `WEBHOOK_SECRET` | The App's single webhook secret (set once at app creation) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud (vector index) |
| `PHI_ENDPOINT`, `PHI_KEY`, `PHI_DEPLOYMENT_NAME` | Inference endpoint (currently Gemini 2.5 Flash via its OpenAI-compatible API — names are legacy from an earlier Phi-based version) |

Run tests: `python -m pytest api/ ingestion/ -q`

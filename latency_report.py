#!/usr/bin/env python3
"""
Pattern Buddy — latency report.

Reconstructs per-stage and end-to-end review latency from the backend's own
telemetry. The App Service emits, per review `trace_id`, structured log lines
(Task 5.1 / 5.2):

    stage_log      {stage, status, latency_ms, metadata}
                   stages: diff_fetch, embedding, qdrant_retrieval, inference_handoff
    system_metrics {end_to_end_latency_ms, inference_latency_ms, ram_mb, status}

Those land in the container stdout logs (LogFiles/*containerStream.log), which
this script pulls via `az webapp log download` and parses. Optionally it also
computes the GitHub-side numbers (trigger overhead + wall-clock) from PR
timestamps so you get the full "PR opened -> comment posted" picture.

Usage
-----
    # Pull a fresh log snapshot from Azure and print the report:
    python latency_report.py --download

    # Parse an already-extracted log directory instead of downloading:
    python latency_report.py --logs /path/to/LogFiles

    # Add GitHub trigger overhead + wall-clock for one review:
    python latency_report.py --download --repo P-dilasha-004/test-repo-codebase-aware-reviewer --pr 8

Requirements
------------
- `az` CLI logged in (for --download).
- For --repo/--pr enrichment: a GitHub PAT in the `PAT` env var (or a .env with
  `PAT=...` next to this script).

Config (override via flags or env):
    APP  = pattern-buddy-v2         (--app / PATTERN_BUDDY_APP)
    RG   = pattern-buddy-rg         (--rg  / PATTERN_BUDDY_RG)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict

APP_DEFAULT = os.environ.get("PATTERN_BUDDY_APP", "pattern-buddy-v2")
RG_DEFAULT  = os.environ.get("PATTERN_BUDDY_RG",  "pattern-buddy-rg")

REVIEW_STAGES = ["diff_fetch", "embedding", "qdrant_retrieval", "inference_handoff"]
_JSON_RE = re.compile(r"\{.*\}")
_TS_RE   = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)")


# ── log acquisition ───────────────────────────────────────────────────────────

def download_logs(app: str, rg: str, dest: str) -> str:
    """Download + extract the App Service log filesystem; return the LogFiles dir."""
    zip_path = os.path.join(dest, "azlogs.zip")
    print(f"  downloading logs for {app} ...", file=sys.stderr)
    subprocess.run(
        ["az", "webapp", "log", "download", "--name", app,
         "--resource-group", rg, "--log-file", zip_path],
        check=True, capture_output=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return os.path.join(dest, "LogFiles")


def _stream_lines(logs_dir: str):
    for f in sorted(glob.glob(os.path.join(logs_dir, "*containerStream.log"))):
        try:
            with open(f, errors="replace") as fh:
                yield from fh
        except OSError:
            continue


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_reviews(logs_dir: str) -> dict[str, dict]:
    """
    Group telemetry by trace_id. Returns {trace_id: {stages: {name: ms},
    end_to_end_ms, inference_ms, ram_mb, status, review_chars, chunks}}.
    """
    reviews: dict[str, dict] = defaultdict(lambda: {"stages": {}})
    for line in _stream_lines(logs_dir):
        if "stage_log " not in line and "system_metrics " not in line:
            continue
        m = _JSON_RE.search(line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        tid = d.get("trace_id")
        if not tid:
            continue
        r = reviews[tid]
        if "stage" in d:  # stage_log
            if d.get("status") == "success":
                r["stages"][d["stage"]] = d.get("latency_ms")
            meta = d.get("metadata", {})
            if "review_chars" in meta:
                r["review_chars"] = meta["review_chars"]
            if "chunks_retrieved" in meta:
                r["chunks"] = meta["chunks_retrieved"]
        else:  # system_metrics
            r["end_to_end_ms"] = d.get("end_to_end_latency_ms")
            r["inference_ms"]  = d.get("inference_latency_ms")
            r["ram_mb"]        = d.get("ram_mb")
            r["status"]        = d.get("status")
    return dict(reviews)


# ── stats ─────────────────────────────────────────────────────────────────────

def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _fmt_ms(ms) -> str:
    return "-" if ms is None else (f"{ms/1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms")


def print_report(reviews: dict[str, dict]) -> None:
    succeeded = {t: r for t, r in reviews.items() if r.get("status") == "success"}
    print(f"\nParsed {len(reviews)} reviews ({len(succeeded)} succeeded).\n")

    if not succeeded:
        print("No successful reviews with system_metrics found in these logs.")
        return

    # Per-review table
    print(f"{'trace_id (pr)':<14}{'diff':>8}{'embed':>8}{'retr':>8}{'infer':>9}"
          f"{'end2end':>9}{'ram':>8}{'chunks':>8}")
    for tid, r in sorted(succeeded.items(), key=lambda kv: kv[0]):
        pr = tid.rsplit("_", 2)[-2] if "_" in tid else "?"
        st = r["stages"]
        print(f"pr#{pr:<11}"
              f"{_fmt_ms(st.get('diff_fetch')):>8}"
              f"{_fmt_ms(st.get('embedding')):>8}"
              f"{_fmt_ms(st.get('qdrant_retrieval')):>8}"
              f"{_fmt_ms(st.get('inference_handoff')):>9}"
              f"{_fmt_ms(r.get('end_to_end_ms')):>9}"
              f"{(str(round(r['ram_mb']))+'MB') if r.get('ram_mb') else '-':>8}"
              f"{r.get('chunks','-'):>8}")

    # Aggregates
    def col(key, stage=False):
        return [(r["stages"].get(key) if stage else r.get(key))
                for r in succeeded.values()
                if (r["stages"].get(key) if stage else r.get(key)) is not None]

    print("\nAggregates (mean / p50 / p95):")
    for label, vals in [
        ("diff_fetch",        col("diff_fetch", True)),
        ("embedding",         col("embedding", True)),
        ("qdrant_retrieval",  col("qdrant_retrieval", True)),
        ("inference_handoff", col("inference_handoff", True)),
        ("end_to_end (backend)", col("end_to_end_ms")),
    ]:
        if vals:
            print(f"  {label:<22} {_fmt_ms(statistics.mean(vals)):>9} / "
                  f"{_fmt_ms(_pct(vals,0.5)):>9} / {_fmt_ms(_pct(vals,0.95)):>9}   (n={len(vals)})")


# ── GitHub enrichment (optional) ──────────────────────────────────────────────

def github_timing(repo: str, pr: int, logs_dir: str) -> None:
    """Print trigger overhead (PR open -> webhook) and wall-clock (PR open -> comment)."""
    import datetime

    pat = os.environ.get("PAT")
    if not pat:
        _load_dotenv()
        pat = os.environ.get("PAT")
    if not pat:
        print("\n(skip GitHub timing: no PAT in env/.env)")
        return

    def api(path):
        # curl rather than urllib: portable TLS (avoids macOS urllib CA issues),
        # and curl is already the project's HTTP tool of choice.
        out = subprocess.run(
            ["curl", "-s", "-H", f"Authorization: Bearer {pat}",
             "-H", "Accept: application/vnd.github+json",
             f"https://api.github.com/repos/{repo}/{path}"],
            check=True, capture_output=True, text=True,
        ).stdout
        return json.loads(out)

    def ts(s):
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

    pr_obj = api(f"pulls/{pr}")
    opened = ts(pr_obj["created_at"])
    comments = api(f"issues/{pr}/comments")
    pb = [c for c in comments]  # first comment is Pattern Buddy's review
    if not pb:
        print(f"\n(no review comment on PR #{pr} yet)")
        return
    posted = ts(pb[-1]["created_at"])

    # webhook-received time = review_started log line for this pr
    started = None
    pat_re = re.compile(rf"review_started repo={re.escape(repo)} pr={pr}\b")
    for line in _stream_lines(logs_dir):
        if pat_re.search(line):
            tm = _TS_RE.match(line)
            if tm:
                started = ts(tm.group(1) if tm.group(1).endswith("Z") else tm.group(1) + "Z")
    print(f"\nGitHub timing for PR #{pr}:")
    print(f"  PR opened            {opened.isoformat()}")
    if started:
        print(f"  webhook received     {started.isoformat()}   (Action trigger overhead "
              f"{(started-opened).total_seconds():.1f}s)")
    print(f"  review comment posted{posted.isoformat()}   (wall-clock "
          f"{(posted-opened).total_seconds():.1f}s)")


def _load_dotenv():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pattern Buddy latency report.")
    ap.add_argument("--download", action="store_true", help="pull fresh logs from Azure")
    ap.add_argument("--logs", help="use an already-extracted LogFiles dir instead")
    ap.add_argument("--app", default=APP_DEFAULT)
    ap.add_argument("--rg",  default=RG_DEFAULT)
    ap.add_argument("--repo", help="owner/repo for GitHub timing enrichment")
    ap.add_argument("--pr", type=int, help="PR number for GitHub timing enrichment")
    args = ap.parse_args()

    tmp = None
    if args.logs:
        logs_dir = args.logs
    elif args.download:
        tmp = tempfile.mkdtemp(prefix="pb_latency_")
        logs_dir = download_logs(args.app, args.rg, tmp)
    else:
        ap.error("pass --download to fetch logs, or --logs <dir> to reuse a snapshot")

    reviews = parse_reviews(logs_dir)
    print_report(reviews)
    if args.repo and args.pr:
        github_timing(args.repo, args.pr, logs_dir)


if __name__ == "__main__":
    main()

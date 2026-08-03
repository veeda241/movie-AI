"""Smoke-check the movie/video pipeline end to end.

Run with::

    python scripts/check_pipeline.py

It will:
  1. Verify the FastAPI server is reachable on ``$MOVIE_FLOW_API_URL``.
  2. Hit ``/health`` to confirm HF_TOKEN / CUDA / video-model env status.
  3. Hit ``/auth/me`` with ``$MOVIE_FLOW_API_TOKEN`` to confirm auth works.
  4. Print the most likely fixes for the most common "video not generating"
     failure modes.

Exits non-zero if anything is obviously broken.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Force UTF-8 stdout so checkmarks/arrows render on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _get(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, str(exc)


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    api_url = os.environ.get("MOVIE_FLOW_API_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.environ.get("MOVIE_FLOW_API_TOKEN", "").strip()

    failures: list[str] = []

    _section("1. API reachability")
    status, body = _get(f"{api_url}/health")
    if status == 0:
        print(f"  ✗ Cannot reach {api_url}: {body}")
        print("    Fix: start the API with `scripts\\run_api.bat` or `uvicorn api.main:app`.")
        failures.append("api unreachable")
    elif status != 200:
        print(f"  ✗ /health returned HTTP {status}: {body[:300]}")
        failures.append(f"health http {status}")
    else:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        print(f"  ✓ {api_url}/health OK")
        for key, value in data.items():
            print(f"    {key:18s} {value}")

        hf_token = str(data.get("hf_token", "")).lower()
        cuda = bool(data.get("cuda", False))
        local_fallback = str(data.get("local_fallback", "true")).lower() in {"true", "1", "yes"}

        if hf_token in {"missing", "", "none"}:
            print("  ⚠ HF_TOKEN is missing — multi-agent movie still works (local planning),")
            print("    but remote video generation will be skipped.")
            print("    Fix: create a free token at https://huggingface.co/settings/tokens")
            print("    and add `HF_TOKEN=hf_xxx` to .env, then restart the API.")
        if not cuda:
            print("  ⚠ CUDA not available — the local Wan 2.1 1.3B model requires a GPU.")
            print("    Pick 'wan-2.2' or 'motif-local' in the UI to use remote/local CPU paths.")
        if not local_fallback:
            print("  ⚠ HF_ALLOW_LOCAL_FALLBACK=false — any remote video failure will surface")
            print("    as a job error instead of a placeholder clip. This is intentional but")
            print("    easy to mistake for a bug.")

    _section("2. Auth check")
    if not token:
        print("  ⚠ MOVIE_FLOW_API_TOKEN is not set — skipping auth check.")
        print("    Get a token from /auth/login and export MOVIE_FLOW_API_TOKEN before re-running.")
    else:
        status, body = _get(f"{api_url}/auth/me", headers={"Authorization": f"Bearer {token}"})
        if status == 200:
            print(f"  ✓ Auth OK ({body[:120]}...)")
        elif status == 401:
            print(f"  ✗ Auth rejected (401). Token expired or invalid: {body[:120]}")
            failures.append("auth 401")
        else:
            print(f"  ✗ Auth check returned HTTP {status}: {body[:200]}")
            failures.append(f"auth http {status}")

    _section("3. Most common 'video not generating' causes")
    fixes = [
        (
            "API server isn't running",
            "Start it: scripts\\run_api.bat   (or: uvicorn api.main:app --reload --port 8000)",
        ),
        (
            "Movie job hangs >5 min",
            "If HF_TOKEN isn't set, the agents fall back to local planning (fast). "
            "If it IS set, each of 5 LLM agents can take 30-90s on HF Inference; "
            "expect 5-10 min total before videos appear.",
        ),
        (
            "wan-2.1-1.3b hangs on first run",
            "First invocation downloads ~2.5 GB of weights. Watch the API console; "
            "downloads are silent unless HF_HUB_ENABLE_HF_TRANSFER=1.",
        ),
        (
            "Output is a colored gradient with 'PLACEHOLDER' text",
            "Remote video failed (model too big, network, missing token). "
            "Set HF_TOKEN and pick a free-tier-compatible model "
            "(HF_VIDEO_MODEL=ali-vilab/text-to-video-ms-1.7b).",
        ),
        (
            "Job stuck in 'queued' or 'running' forever",
            "Background worker crashed. Check API logs for traceback. "
            "The latest code marks crashed jobs as 'failed' automatically.",
        ),
    ]
    for title, fix in fixes:
        print(f"  • {title}")
        print(f"    → {fix}")

    _section("Summary")
    if failures:
        print(f"  ✗ {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("  ✓ No structural issues detected. If videos still don't render, paste the")
    print("    full output of this script + the failing job's events_json into a bug report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download linoyts Wan action datasets into data/video_lab/raw/.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\download_hf_wan_actions.py
  .\\.venv\\Scripts\\python.exe scripts\\download_hf_wan_actions.py --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF Wan action MP4 datasets")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only first N repos (0 = all). Useful for a smoke test.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Specific repo id (repeatable). Default: full linoyts list.",
    )
    args = parser.parse_args()

    from video_lab.data.hf_wan_datasets import (
        DEFAULT_WAN_ACTION_DATASETS,
        download_all_wan_action_datasets,
    )

    repos: list[str]
    if args.repo:
        repos = list(args.repo)
    else:
        repos = list(DEFAULT_WAN_ACTION_DATASETS)
        if args.limit and args.limit > 0:
            repos = repos[: int(args.limit)]

    summary = download_all_wan_action_datasets(repos)
    print(summary)
    print("Next: Gradio Data -> Curate -> Recaption -> Train (match niche to these actions)")
    return 0 if int(summary.get("downloaded") or 0) > 0 or int(summary.get("skipped") or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

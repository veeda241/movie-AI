"""Download Pexels stock videos into data/video_lab/raw/ for Video Lab training.

Videos provided by Pexels — https://www.pexels.com

Usage:
  set PEXELS_API_KEY in .env
  python scripts/download_pexels.py --query "ocean waves" --count 200
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
    parser = argparse.ArgumentParser(description="Download Pexels videos for Video Lab training")
    parser.add_argument("--query", required=True, help='Niche query, e.g. "neon city night"')
    parser.add_argument("--count", type=int, default=200, help="How many clips to download (default 200)")
    parser.add_argument("--min-duration", type=int, default=3)
    parser.add_argument("--max-duration", type=int, default=15)
    parser.add_argument("--orientation", default="landscape", choices=["landscape", "portrait", "square"])
    parser.add_argument("--size", default="medium", choices=["large", "medium", "small"], help="API size filter")
    parser.add_argument("--max-width", type=int, default=1280, help="Prefer MP4 near this width")
    args = parser.parse_args()

    from video_lab.data.pexels_download import download_pexels_videos

    summary = download_pexels_videos(
        args.query,
        target_count=int(args.count),
        min_duration=int(args.min_duration),
        max_duration=int(args.max_duration),
        orientation=args.orientation,
        size=args.size,
        prefer_max_width=int(args.max_width),
    )
    print(summary)
    print("Next: open Gradio Data tab → Curate raw → Recaption → Train")
    return 0 if int(summary.get("downloaded") or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""CogVideoX-5B LoRA fine-tune on HF Wan action clips.

Builds manifest from downloaded clips, then runs LoRA training.

Examples (PowerShell):
  .venv/Scripts/python.exe scripts/train_lora.py --steps 500 --rank 16
  .venv/Scripts/python.exe scripts/train_lora.py --steps 200 --rank 8 --base-model THUDM/CogVideoX-5b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from video_lab.utils.regex_shim import ensure_regex_shim

    ensure_regex_shim()
except Exception:
    pass

from video_lab import RAW_DIR, DATA_ROOT
from video_lab.data.hf_wan_datasets import build_hf_wan_manifest
from video_lab.train.train_lora_t2v import train_lora_t2v
from video_lab.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="CogVideoX-5B LoRA fine-tune on HF Wan actions")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps (default: 200)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--base-model", type=str, default="THUDM/CogVideoX-5b", help="Base model ID")
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional pre-built manifest path")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Force rebuild manifest from raw clips")
    args = parser.parse_args()

    device = get_device()
    print(f"device={device}")
    print(f"Steps={args.steps} Rank={args.rank} LR={args.lr}")
    print(f"Base model: {args.base_model}")

    # ------------------------------------------------------------------ #
    # 1. Build manifest from HF Wan clips (if not provided)
    # ------------------------------------------------------------------ #
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        print(f"Using provided manifest: {manifest_path}")
    else:
        manifest_path = DATA_ROOT / "manifest_hf_wan.jsonl"
        if not manifest_path.exists() or args.rebuild_manifest:
            print("Building manifest from HF Wan clips...")
            manifest_path = build_hf_wan_manifest(manifest_path=manifest_path, raw_dir=RAW_DIR)
            if not manifest_path.exists():
                print(f"ERROR: No HF Wan clips found in {RAW_DIR}")
                print("Run scripts/download_hf_wan_actions.py first to download clips.")
                sys.exit(1)
        else:
            print(f"Using existing manifest: {manifest_path}")

    # Count clips
    count = sum(1 for _ in open(manifest_path, encoding="utf-8") if _.strip())
    print(f"Manifest has {count} clips")

    # ------------------------------------------------------------------ #
    # 2. Run LoRA training
    # ------------------------------------------------------------------ #
    def log(msg: str) -> None:
        print(msg, flush=True)

    print("\n--- Starting LoRA training ---")
    result = train_lora_t2v(
        manifest_path=manifest_path,
        base_model=args.base_model,
        steps=args.steps,
        rank=args.rank,
        lr=args.lr,
        log_fn=log,
    )
    print(f"\nLoRA adapter saved to: {result}")
    print("Done. Launch Gradio and use the Generate tab with Fine-tune model to test.")


if __name__ == "__main__":
    main()

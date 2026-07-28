#!/usr/bin/env python
r"""Wan2.1-T2V-1.3B LoRA fine-tune on HF Wan action clips.

Examples (PowerShell) — 16GB GPU (recommended):
  .\.venv\Scripts\python.exe scripts\train_lora.py --steps 1000 --rank 16 --low-vram

  # Smaller size for tight VRAM:
  .\.venv\Scripts\python.exe scripts\train_lora.py --steps 1000 --rank 8 --height 256 --width 256 --frames 17 --low-vram

  # 24GB+ GPU (full native 832x480x81):
  .\.venv\Scripts\python.exe scripts\train_lora.py --steps 1000 --rank 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from video_lab.utils.regex_shim import ensure_regex_shim

    ensure_regex_shim()
except Exception:
    pass

from video_lab import RAW_DIR, DATA_ROOT
from video_lab.config import LabConfig
from video_lab.data.hf_wan_datasets import build_hf_wan_manifest
from video_lab.train.train_lora_t2v import _wan_frames, _wan_image_size, train_lora_t2v
from video_lab.utils.device import get_device


def main() -> None:
    cfg = LabConfig()
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B LoRA fine-tune on HF Wan actions")
    parser.add_argument("--steps", type=int, default=200, help="Training steps (default: 200)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--base-model",
        type=str,
        default=cfg.base_t2v_model,
        help=f"Base model ID (default: {cfg.base_t2v_model} — diffusers-format Wan2.1-1.3B)",
    )
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--height", type=int, default=cfg.wan_height, help=f"Frame height, must be /16 (default {cfg.wan_height})")
    parser.add_argument("--width", type=int, default=cfg.wan_width, help=f"Frame width, must be /16 (default {cfg.wan_width})")
    parser.add_argument(
        "--frames",
        type=int,
        default=cfg.wan_frames,
        help=f"Must be 4N+1 for Wan (5, 9, 13, ... 81). Default {cfg.wan_frames} (native).",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=None,
        help="Force CPU offload for umt5/VAE (auto-on if GPU < 20GB)",
    )
    parser.add_argument("--no-low-vram", action="store_true", help="Keep umt5/VAE on GPU")
    args = parser.parse_args()

    low_vram: bool | None
    if args.no_low_vram:
        low_vram = False
    elif args.low_vram:
        low_vram = True
    else:
        low_vram = None  # auto

    height, width = _wan_image_size(args.height, args.width)
    frames = _wan_frames(args.frames)

    device = get_device()
    print(f"device={device}")
    print(f"Steps={args.steps} Rank={args.rank} LR={args.lr}")
    print(f"Base model: {args.base_model}")
    print(f"Size={width}x{height} frames={frames} low_vram={low_vram}")

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        print(f"Using provided manifest: {manifest_path}")
    else:
        manifest_path = DATA_ROOT / "manifest_hf_wan.jsonl"
        if not manifest_path.exists() or args.rebuild_manifest:
            print("Building manifest from HF Wan clips...")
            manifest_path = build_hf_wan_manifest(manifest_path=manifest_path, raw_dir=RAW_DIR)
            if not manifest_path.exists() or manifest_path.stat().st_size == 0:
                print(f"ERROR: No HF Wan clips found in {RAW_DIR}")
                print("Run scripts/download_hf_wan_actions.py first.")
                sys.exit(1)
        else:
            print(f"Using existing manifest: {manifest_path}")

    count = sum(1 for _ in open(manifest_path, encoding="utf-8") if _.strip())
    print(f"Manifest has {count} clips")

    def log(msg: str) -> None:
        print(msg, flush=True)

    print("\n--- Starting Wan LoRA training ---")
    result = train_lora_t2v(
        manifest_path=manifest_path,
        base_model=args.base_model,
        steps=args.steps,
        rank=args.rank,
        lr=args.lr,
        height=height,
        width=width,
        frames=frames,
        low_vram=low_vram,
        log_fn=log,
    )
    print(f"\nLoRA adapter saved to: {result}")
    print("Done. Gradio -> Experimental (Wan LoRA) -> Generate with Wan.")


if __name__ == "__main__":
    main()

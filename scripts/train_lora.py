#!/usr/bin/env python
"""CogVideoX LoRA fine-tune on HF Wan action clips.

Examples (PowerShell) — 16GB GPU (recommended):
  .\\.venv\\Scripts\\python.exe scripts\\train_lora.py --base-model THUDM/CogVideoX-2b --steps 1000 --rank 16 --low-vram

  # CogVideoX-5b on 16GB (tight; uses CPU VAE/text offload):
  .\\.venv\\Scripts\\python.exe scripts\\train_lora.py --base-model THUDM/CogVideoX-5b --steps 1000 --rank 8 --height 192 --width 192 --low-vram

  # 24GB+ GPU:
  .\\.venv\\Scripts\\python.exe scripts\\train_lora.py --base-model THUDM/CogVideoX-5b --steps 1000 --rank 16
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
from video_lab.data.hf_wan_datasets import build_hf_wan_manifest
from video_lab.train.train_lora_t2v import train_lora_t2v
from video_lab.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="CogVideoX LoRA fine-tune on HF Wan actions")
    parser.add_argument("--steps", type=int, default=200, help="Training steps (default: 200)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--base-model",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Base model ID (default: CogVideoX-2b for 16GB GPUs; use 5b on 24GB+)",
    )
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="Must be 8N+1 for CogVideoX (9, 17, 25, 49). Default 9 for 16GB.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=None,
        help="Force CPU offload for VAE/text (auto-on if GPU < 20GB)",
    )
    parser.add_argument("--no-low-vram", action="store_true", help="Keep VAE/text on GPU")
    args = parser.parse_args()

    low_vram: bool | None
    if args.no_low_vram:
        low_vram = False
    elif args.low_vram:
        low_vram = True
    else:
        low_vram = None  # auto

    device = get_device()
    print(f"device={device}")
    print(f"Steps={args.steps} Rank={args.rank} LR={args.lr}")
    print(f"Base model: {args.base_model}")
    print(f"Size={args.width}x{args.height} frames={args.frames} low_vram={low_vram}")

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

    print("\n--- Starting LoRA training ---")
    result = train_lora_t2v(
        manifest_path=manifest_path,
        base_model=args.base_model,
        steps=args.steps,
        rank=args.rank,
        lr=args.lr,
        height=args.height,
        width=args.width,
        frames=args.frames,
        low_vram=low_vram,
        log_fn=log,
    )
    print(f"\nLoRA adapter saved to: {result}")
    print("Done. Gradio → Experimental (CogVideo LoRA) → Generate with LoRA.")


if __name__ == "__main__":
    main()

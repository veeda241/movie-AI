#!/usr/bin/env python
"""Train niche-profile VAE then DiT (laptop 256² or 24GB cloud).

Examples (PowerShell):
  .\\.venv\\Scripts\\python.exe scripts\\train_niche.py --profile niche_laptop
  .\\.venv\\Scripts\\python.exe scripts\\train_niche.py --profile niche_24gb --vae-steps 2000 --dit-steps 4000
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

try:
    from video_lab.utils.regex_shim import ensure_regex_shim

    ensure_regex_shim()
except Exception:
    pass

from video_lab import MANIFEST_PATH, RAW_DIR
from video_lab.data.recaption import recaption_manifest
from video_lab.data.smoke import ensure_smoke_manifest
from video_lab.train.niche_profile import niche_train_kwargs
from video_lab.train.train_dit import train_dit
from video_lab.train.train_vae import train_vae
from video_lab.utils.device import get_device


def _count_raw_mp4() -> int:
    if not RAW_DIR.exists():
        return 0
    return sum(1 for p in RAW_DIR.glob("*.mp4") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Niche VAE+DiT training")
    parser.add_argument(
        "--profile",
        default="niche_laptop",
        choices=["niche_laptop", "niche_24gb", "niche_24gb_512"],
        help="Training profile",
    )
    parser.add_argument("--vae-steps", type=int, default=None)
    parser.add_argument("--dit-steps", type=int, default=None)
    parser.add_argument("--refresh-smoke", action="store_true", help="Rebuild smoke clips at target size")
    args = parser.parse_args()

    device = get_device()
    kw = niche_train_kwargs(args.profile)
    profile = kw["profile"]
    print(f"device={device} profile={profile.name} min_vram~={profile.min_vram_gb}GB", flush=True)
    print(profile.description, flush=True)
    print(f"target {kw['width']}x{kw['height']} x {kw['frames']}f dit={kw['dit_size']}", flush=True)

    n_raw = _count_raw_mp4()
    print(f"raw clips in {RAW_DIR}: {n_raw}", flush=True)
    print(f"manifest: {MANIFEST_PATH}", flush=True)

    if args.refresh_smoke or not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0:
        if n_raw == 0:
            print(
                "WARNING: no MP4s in data/video_lab/raw/ — creating smoke dataset only. "
                "Run scripts/download_hf_wan_actions.py or download_pexels.py first for real training.",
                flush=True,
            )
        ensure_smoke_manifest(
            MANIFEST_PATH,
            frames=int(kw["frames"]),
            size=max(int(kw["height"]), int(kw["width"])),
        )
        recaption_manifest()
        print(f"smoke/manifest ready: {MANIFEST_PATH}", flush=True)
    elif n_raw == 0:
        print(
            "WARNING: manifest exists but raw/ has 0 MP4s — training may use missing paths / smoke fills.",
            flush=True,
        )

    vae_steps = args.vae_steps or int(kw["vae_steps"])
    dit_steps = args.dit_steps or int(kw["dit_steps"])

    def log(msg: str) -> None:
        print(msg, flush=True)

    train_vae(
        steps=vae_steps,
        frames=int(kw["frames"]),
        height=int(kw["height"]),
        width=int(kw["width"]),
        bucket=kw["bucket"],
        min_aesthetic=float(kw["min_aesthetic"]),
        train_stage=kw["train_stage"],
        use_amp=True,
        log_fn=log,
    )
    train_dit(
        steps=dit_steps,
        frames=int(kw["frames"]),
        height=int(kw["height"]),
        width=int(kw["width"]),
        bucket=kw["bucket"],
        min_aesthetic=float(kw["min_aesthetic"]),
        dit_size=kw["dit_size"],
        train_stage=kw["train_stage"],
        use_amp=True,
        log_fn=log,
    )
    print("Done. Generate at matching frames/resolution in Gradio (256 / ~0.7s @ 12fps for niche_laptop).", flush=True)


if __name__ == "__main__":
    main()

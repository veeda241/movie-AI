#!/usr/bin/env python
"""Smoke-check that Movie-AI lab + API import and core paths exist."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    errors: list[str] = []

    required = [
        ROOT / "gradio_video_lab.py",
        ROOT / "requirements-video-lab.txt",
        ROOT / "video_lab" / "app.py",
        ROOT / "api" / "main.py",
        ROOT / "scripts" / "train_niche.py",
        ROOT / "scripts" / "download_hf_wan_actions.py",
        ROOT / "scripts" / "download_pexels.py",
        ROOT / "README.md",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing file: {path.relative_to(ROOT)}")

    mods = [
        "video_lab.app",
        "video_lab.infer.research_generate",
        "video_lab.train.train_vae",
        "video_lab.train.train_dit",
        "video_lab.data.curate",
        "video_lab.data.hf_wan_datasets",
        "video_lab.data.pexels_download",
        "api.main",
    ]
    for name in mods:
        try:
            __import__(name)
            print(f"OK  import {name}")
        except Exception as exc:
            errors.append(f"import {name}: {exc}")
            print(f"FAIL import {name}: {exc}")

    try:
        from video_lab.app import build_app

        build_app()
        print("OK  build_app()")
    except Exception as exc:
        errors.append(f"build_app: {exc}")
        print(f"FAIL build_app: {exc}")

    try:
        from video_lab.config import LabConfig

        cfg = LabConfig()
        assert cfg.height == 256 and cfg.frames == 8, "LabConfig should default to niche_laptop-like 256/8f"
        print(f"OK  LabConfig defaults {cfg.width}x{cfg.height} f{cfg.frames}")
    except Exception as exc:
        errors.append(f"LabConfig: {exc}")
        print(f"FAIL LabConfig: {exc}")

    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

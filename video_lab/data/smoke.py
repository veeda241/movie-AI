from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from video_lab import SMOKE_DIR, ensure_dirs
from video_lab.data.dataset import write_manifest
from video_lab.utils.video_io import write_rgb_video


CAPTIONS = [
    "blue particles drifting over a dark gradient",
    "warm orange light sweeping left to right",
    "green ripples expanding from the center",
    "violet glow pulsing softly in the frame",
]


def render_smoke_clip(
    index: int,
    *,
    frames: int = 24,
    size: int = 128,
) -> np.ndarray:
    """Structured motion patterns — easy for a tiny VAE/DiT to learn."""
    arr = np.zeros((frames, size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    for t in range(frames):
        u = t / max(frames - 1, 1)
        if index % 4 == 0:
            # Blue particles / diagonal drift
            phase = (xx + yy) / size + u * 2.5
            r = 20 + 40 * (0.5 + 0.5 * np.sin(phase * 4))
            g = 30 + 50 * (0.5 + 0.5 * np.cos(phase * 3))
            b = 80 + 140 * (0.5 + 0.5 * np.sin(phase * 5 + 1))
            dots = ((np.sin(xx * 0.35 + t * 0.8) * np.sin(yy * 0.35 - t * 0.6)) > 0.7).astype(np.float32)
            r = r * (0.55 + 0.45 * dots)
            g = g * (0.55 + 0.45 * dots)
            b = np.clip(b + 80 * dots, 0, 255)
        elif index % 4 == 1:
            # Warm orange sweep L→R
            band = np.clip(1.0 - np.abs((xx / size) - u) * 3.5, 0, 1)
            base = 0.15 + 0.1 * (yy / size)
            r = (40 + 200 * band + 30 * base) 
            g = (20 + 90 * band + 20 * base)
            b = (10 + 25 * band)
        elif index % 4 == 2:
            # Green ripples from center
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size
            wave = 0.5 + 0.5 * np.sin(dist * 18 - u * 10)
            r = 15 + 40 * wave
            g = 60 + 160 * wave
            b = 30 + 50 * wave
        else:
            # Violet pulse
            pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(u * math.pi * 2))
            rad = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.28) ** 2))
            r = 40 + 160 * pulse * rad
            g = 20 + 40 * pulse * rad
            b = 80 + 160 * pulse * rad
        arr[t] = np.stack(
            [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)],
            axis=-1,
        ).astype(np.uint8)
    return arr


def make_smoke_clips(n: int = 24, frames: int = 24, size: int = 128) -> list[dict]:
    from video_lab.data.recaption import densify_row

    ensure_dirs()
    rows: list[dict] = []
    for i in range(n):
        out = SMOKE_DIR / f"smoke_{i:02d}.mp4"
        arr = render_smoke_clip(i, frames=frames, size=size)
        write_rgb_video(arr, out, fps=12)
        row = {
            "path": str(out.resolve()),
            "caption": CAPTIONS[i % len(CAPTIONS)],
            "camera": "static",
            "lighting": ["cool", "warm", "daylight", "magenta"][i % 4],
            "motion": ["particles", "sweep", "ripples", "pulse"][i % 4],
            "aesthetic": 6,
            "tags": ["smoke", "abstract"],
            "negative": "photoreal faces, text overlay",
            "fps": 12,
            "frames": frames,
            "width": size,
            "height": size,
            "bucket": "square_128" if size >= 128 else "square_96",
            "bucket_w": size,
            "bucket_h": size,
            "flow_mean": 1.0,
            "flow_var": 0.2,
        }
        rows.append(densify_row(row, fill_empty_labels=False))
    return rows


def ensure_smoke_manifest(manifest_path: Path, *, frames: int = 24, size: int = 128, n: int = 24) -> Path:
    rows = make_smoke_clips(n=n, frames=frames, size=size)
    return write_manifest(rows, manifest_path)

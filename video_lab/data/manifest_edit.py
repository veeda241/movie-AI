"""Edit / suggest manual feature fields on manifest.jsonl."""

from __future__ import annotations

import json
import re
from pathlib import Path

from video_lab import MANIFEST_PATH
from video_lab.data.dataset import write_manifest

CAMERA_CHOICES = ["", "static", "pan left", "pan right", "tilt up", "tilt down", "zoom in", "zoom out", "dolly forward", "handheld"]
LIGHTING_CHOICES = ["", "daylight", "golden hour", "neon night", "cool", "warm", "low-key", "overcast", "magenta"]
MOTION_CHOICES = ["", "static subject", "particles", "sweep", "ripples", "pulse", "walk", "drive-by", "camera only"]


def load_rows(path: Path | None = None) -> list[dict]:
    path = path or MANIFEST_PATH
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def save_rows(rows: list[dict], path: Path | None = None) -> Path:
    return write_manifest(rows, path or MANIFEST_PATH)


def row_labels(rows: list[dict] | None = None) -> list[str]:
    rows = rows if rows is not None else load_rows()
    labels: list[str] = []
    for i, row in enumerate(rows):
        stem = Path(str(row.get("path", f"row_{i}"))).name
        cap = str(row.get("caption", ""))[:48]
        labels.append(f"{i}: {stem} — {cap}")
    return labels or ["(empty manifest)"]


def get_row(index: int, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else load_rows()
    if not rows:
        return {
            "path": "",
            "caption": "",
            "camera": "",
            "lighting": "",
            "motion": "",
            "aesthetic": 5,
            "tags": [],
            "negative": "",
            "fps": 12,
            "frames": 24,
        }
    return dict(rows[index % len(rows)])


def suggest_from_caption(caption: str) -> dict:
    """Heuristic label suggestions from caption text (manual override still wins)."""
    text = (caption or "").lower()
    camera = ""
    for key, val in [
        ("dolly", "dolly forward"),
        ("handheld", "handheld"),
        ("zoom in", "zoom in"),
        ("zoom out", "zoom out"),
        ("pan left", "pan left"),
        ("pan right", "pan right"),
        ("tilt", "tilt up"),
        ("slow pan", "pan left"),
        ("static", "static"),
    ]:
        if key in text:
            camera = val
            break
    if not camera and any(k in text for k in ("sweep", "drifting", "pulse", "ripple")):
        camera = "static"

    lighting = ""
    for key, val in [
        ("neon", "neon night"),
        ("golden", "golden hour"),
        ("sunset", "golden hour"),
        ("night", "neon night"),
        ("warm", "warm"),
        ("orange", "warm"),
        ("cool", "cool"),
        ("blue", "cool"),
        ("violet", "magenta"),
        ("magenta", "magenta"),
        ("daylight", "daylight"),
        ("green", "daylight"),
        ("overcast", "overcast"),
        ("low-key", "low-key"),
    ]:
        if key in text:
            lighting = val
            break

    motion = ""
    for key, val in [
        ("particle", "particles"),
        ("drift", "particles"),
        ("sweep", "sweep"),
        ("ripple", "ripples"),
        ("pulse", "pulse"),
        ("walk", "walk"),
        ("drive", "drive-by"),
    ]:
        if key in text:
            motion = val
            break

    tags: list[str] = []
    for tag in ("cinematic", "city", "rain", "nature", "abstract", "smoke", "neon", "night"):
        if tag in text:
            tags.append(tag)
    if not tags and text:
        tags = ["clip"]

    negative = "watermark, blurry, text overlay"
    aesthetic = 6
    if any(k in text for k in ("cinematic", "beautiful", "detailed")):
        aesthetic = 8

    return {
        "camera": camera,
        "lighting": lighting,
        "motion": motion,
        "aesthetic": aesthetic,
        "tags": tags,
        "negative": negative,
    }


def update_row(
    index: int,
    *,
    caption: str,
    camera: str,
    lighting: str,
    motion: str,
    aesthetic: float,
    tags: str,
    negative: str,
    path: Path | None = None,
) -> list[dict]:
    rows = load_rows(path)
    if not rows:
        raise ValueError("Manifest is empty — create smoke or curate raw clips first.")
    idx = index % len(rows)
    row = dict(rows[idx])
    row["caption"] = (caption or "").strip()
    row["camera"] = (camera or "").strip()
    row["lighting"] = (lighting or "").strip()
    row["motion"] = (motion or "").strip()
    row["aesthetic"] = float(aesthetic)
    tag_list = [t.strip() for t in re.split(r"[,|]", tags or "") if t.strip()]
    row["tags"] = tag_list
    row["negative"] = (negative or "").strip()
    from video_lab.data.recaption import densify_row

    rows[idx] = densify_row(row, fill_empty_labels=False)
    save_rows(rows, path)
    return rows


def autofill_empty_labels(path: Path | None = None) -> tuple[int, Path]:
    """Fill blank camera/lighting/motion/tags from caption heuristics."""
    path = path or MANIFEST_PATH
    from video_lab.data.recaption import densify_row

    rows = load_rows(path)
    changed = 0
    new_rows = []
    for row in rows:
        before = dict(row)
        row2 = densify_row(row, fill_empty_labels=True)
        if row2 != before:
            changed += 1
        new_rows.append(row2)
    save_rows(new_rows, path)
    return changed, path

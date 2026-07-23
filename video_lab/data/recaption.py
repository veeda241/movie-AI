"""Offline dense recaptioning from Labels fields + heuristics (no cloud VLM required)."""

from __future__ import annotations

from pathlib import Path

from video_lab import MANIFEST_PATH
from video_lab.data.manifest_edit import load_rows, save_rows, suggest_from_caption


def build_dense_caption(row: dict) -> str:
    """
    Dense cinematography-oriented caption for training.
    Prefers existing dense_caption parts + Labels; fills gaps via heuristics.
    """
    path = Path(str(row.get("path", "clip")))
    base = str(row.get("caption", "")).strip() or path.stem.replace("_", " ").replace("-", " ")
    sug = suggest_from_caption(base)

    camera = str(row.get("camera", "")).strip() or sug.get("camera", "")
    lighting = str(row.get("lighting", "")).strip() or sug.get("lighting", "")
    motion = str(row.get("motion", "")).strip() or sug.get("motion", "")
    tags = row.get("tags") if row.get("tags") else sug.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    parts = [base]
    if camera:
        parts.append(f"Camera: {camera}")
    if lighting:
        parts.append(f"Lighting: {lighting}")
    if motion:
        parts.append(f"Motion: {motion}")
    if tags:
        parts.append("Style tags: " + ", ".join(str(t) for t in tags if t))
    neg = str(row.get("negative", "")).strip()
    if neg:
        parts.append(f"Avoid: {neg}")
    return ". ".join(parts)


def densify_row(row: dict, *, fill_empty_labels: bool = True) -> dict:
    out = dict(row)
    base = str(out.get("caption", "")).strip()
    if fill_empty_labels and base:
        sug = suggest_from_caption(base)
        for key in ("camera", "lighting", "motion", "negative"):
            if not str(out.get(key, "")).strip() and sug.get(key):
                out[key] = sug[key]
        if not out.get("tags") and sug.get("tags"):
            out["tags"] = sug["tags"]
        if not out.get("aesthetic"):
            out["aesthetic"] = sug.get("aesthetic", 5)
    out["dense_caption"] = build_dense_caption(out)
    # Keep training compose helper aligned
    out.setdefault("caption", base or Path(str(out.get("path", "clip"))).stem)
    return out


def recaption_manifest(path: Path | None = None, *, fill_empty_labels: bool = True) -> tuple[int, Path]:
    path = path or MANIFEST_PATH
    rows = load_rows(path)
    updated = 0
    new_rows = []
    for row in rows:
        before = str(row.get("dense_caption", ""))
        row2 = densify_row(row, fill_empty_labels=fill_empty_labels)
        if row2.get("dense_caption") != before:
            updated += 1
        new_rows.append(row2)
    save_rows(new_rows, path)
    return updated, path

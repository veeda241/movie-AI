from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from video_lab.data.buckets import letterbox_frames, parse_bucket_choice
from video_lab.utils.video_io import read_video_frames, tensor_from_frames


def _is_usable_video_path(path: Path) -> bool:
    if not path or str(path) in {"", "SYNTHETIC", "synthetic"}:
        return False
    return path.exists() and path.is_file()


def compose_caption(row: dict, path: Path | None = None) -> str:
    """Prefer dense_caption; else merge Labels into one training caption."""
    dense = str(row.get("dense_caption", "")).strip()
    if dense:
        return dense
    path = path or Path(str(row.get("path", "clip")))
    parts: list[str] = []
    base = str(row.get("caption", "")).strip() or path.stem.replace("_", " ")
    parts.append(base)
    for key in ("camera", "lighting", "motion"):
        val = str(row.get(key, "")).strip()
        if val and val.upper() != "REPLACE" and not val.startswith("REPLACE"):
            parts.append(f"{key}: {val}")
    tags = row.get("tags")
    if isinstance(tags, list) and tags:
        clean = [str(t) for t in tags if t and not str(t).startswith("REPLACE")]
        if clean:
            parts.append("tags: " + ", ".join(clean))
    return ". ".join(parts)


# Back-compat alias
_compose_caption = compose_caption


class VideoManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        *,
        frames: int = 8,
        height: int = 64,
        width: int = 64,
        min_aesthetic: float = 0.0,
        bucket: str | None = None,
        letterbox: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.frames = frames
        self.min_aesthetic = float(min_aesthetic)
        self.letterbox = letterbox
        if bucket:
            b = parse_bucket_choice(bucket)
            self.height, self.width = b.height, b.width
            self.bucket_name = b.name
        else:
            self.height = height
            self.width = width
            self.bucket_name = f"{width}x{height}"
        self.rows: list[dict] = []
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                path = Path(str(row.get("path", "")))
                row_type = str(row.get("type", "video")).lower()
                if row_type == "image" and str(path) == "SYNTHETIC":
                    continue
                try:
                    score = float(row.get("aesthetic", 0) or 0)
                except (TypeError, ValueError):
                    score = 0.0
                if self.min_aesthetic > 0 and score > 0 and score < self.min_aesthetic:
                    continue
                # Optional: if row declares a different bucket and we selected one, soft-prefer match
                if bucket and row.get("bucket") and row.get("bucket") != self.bucket_name:
                    # still include — pad to train bucket; do not drop
                    pass
                self.rows.append(row)

    def __len__(self) -> int:
        return max(len(self.rows), 1)

    def _synthetic_video(self, caption: str, index: int) -> torch.Tensor:
        from video_lab.data.smoke import render_smoke_clip

        arr = render_smoke_clip(index % 4, frames=self.frames, size=max(self.height, self.width))
        if arr.shape[1] != self.height or arr.shape[2] != self.width:
            from PIL import Image

            resized = []
            for frame in arr:
                resized.append(
                    np.asarray(
                        Image.fromarray(frame).resize((self.width, self.height), Image.BILINEAR),
                        dtype=np.uint8,
                    )
                )
            arr = np.stack(resized, axis=0)
        return tensor_from_frames(arr.astype(np.float32) / 255.0).squeeze(0)

    def _load_frames(self, path: Path) -> np.ndarray:
        if self.letterbox:
            frames = read_video_frames(
                path,
                max_frames=self.frames,
                height=self.height,
                width=self.width,
                preserve_aspect=True,
                max_side=max(self.height, self.width),
            )
            frames, _meta = letterbox_frames(frames, self.height, self.width)
            return frames
        return read_video_frames(path, max_frames=self.frames, height=self.height, width=self.width)

    def __getitem__(self, index: int) -> dict:
        if not self.rows:
            video = self._synthetic_video("synthetic smoke clip", index)
            return {"video": video, "caption": "synthetic smoke clip", "path": ""}
        row = self.rows[index % len(self.rows)]
        path = Path(str(row.get("path", "")))
        caption = compose_caption(row, path)
        if not _is_usable_video_path(path):
            video = self._synthetic_video(caption, index)
            return {"video": video, "caption": caption, "path": "SYNTHETIC"}
        frames = self._load_frames(path)
        if float(frames.std()) < 1e-3:
            video = self._synthetic_video(caption, index)
            return {"video": video, "caption": caption, "path": str(path)}
        video = tensor_from_frames(frames).squeeze(0)
        return {"video": video, "caption": caption, "path": str(path)}


def write_manifest(rows: list[dict], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

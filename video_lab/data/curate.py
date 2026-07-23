from __future__ import annotations

import shutil
from pathlib import Path

from video_lab import PROCESSED_DIR, RAW_DIR, ensure_dirs
from video_lab.data.buckets import choose_bucket_for_size
from video_lab.data.dataset import write_manifest
from video_lab.data.optical_flow import flow_filter_clip
from video_lab.data.recaption import densify_row


def ingest_folder(src: str | Path, dest: Path | None = None) -> list[Path]:
    ensure_dirs()
    src = Path(src)
    dest = dest or RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if not src.exists():
        return copied
    for path in src.rglob("*"):
        if path.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
            continue
        target = dest / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        copied.append(target)
    return copied


def scene_cut_file(path: Path, out_dir: Path | None = None) -> list[Path]:
    """Split a video into scenes; falls back to copying whole clip."""
    out_dir = out_dir or (PROCESSED_DIR / path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
        from scenedetect.video_splitter import split_video_ffmpeg

        video = open_video(str(path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()
        if len(scenes) <= 1:
            target = out_dir / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            return [target]
        split_video_ffmpeg(str(path), scenes, str(out_dir), show_progress=False)
        return sorted(out_dir.glob("*.mp4"))
    except Exception:
        target = out_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        return [target]


def motion_score(path: Path, sample_frames: int = 12) -> float:
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        prev = None
        diffs = []
        count = 0
        while count < sample_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            if prev is not None:
                diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))))
            prev = gray
            count += 1
        cap.release()
        if not diffs:
            return 0.0
        return float(sum(diffs) / len(diffs))
    except Exception:
        return 1.0


def probe_video(path: Path) -> dict:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 8)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration = n / max(fps, 1.0)
        return {"fps": fps, "frames": n, "width": w, "height": h, "duration": duration}
    except Exception:
        return {"fps": 8.0, "frames": 0, "width": 0, "height": 0, "duration": 0.0}


def filter_clip(path: Path, *, min_motion: float = 0.5, min_seconds: float = 0.5) -> bool:
    meta = probe_video(path)
    if meta["duration"] < min_seconds or meta["width"] < 32 or meta["height"] < 32:
        return False
    return motion_score(path) >= min_motion


def caption_for_path(path: Path, hf_token: str = "") -> str:
    try:
        from video_lab.data.pexels_download import caption_from_pexels_meta

        pexels_cap = caption_from_pexels_meta(path.name)
        if pexels_cap:
            return pexels_cap
    except Exception:
        pass
    try:
        from video_lab.data.hf_wan_datasets import caption_from_hf_wan

        hf_cap = caption_from_hf_wan(path.name)
        if hf_cap:
            return hf_cap
    except Exception:
        pass
    stem = path.stem.replace("_", " ").replace("-", " ")
    return f"cinematic video clip showing {stem}"


def build_manifest_from_raw(
    raw_dir: Path | None = None,
    *,
    manifest_path: Path,
    run_scene_cut: bool = True,
    min_flow: float = 0.15,
    max_flow: float = 12.0,
    max_flow_var: float = 40.0,
    use_optical_flow: bool = True,
) -> Path:
    ensure_dirs()
    raw_dir = raw_dir or RAW_DIR
    rows: list[dict] = []
    sources = list(raw_dir.glob("*")) if raw_dir.exists() else []
    clips: list[Path] = []
    for src in sources:
        if src.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
            continue
        if run_scene_cut:
            clips.extend(scene_cut_file(src))
        else:
            clips.append(src)

    for clip in clips:
        if not filter_clip(clip):
            continue
        flow_stats = None
        if use_optical_flow:
            ok_flow, flow_stats = flow_filter_clip(
                clip,
                min_flow=min_flow,
                max_flow=max_flow,
                max_flow_var=max_flow_var,
            )
            if not ok_flow:
                continue
        meta = probe_video(clip)
        bucket = choose_bucket_for_size(int(meta["width"]), int(meta["height"]))
        row = {
            "path": str(clip.resolve()),
            "caption": caption_for_path(clip),
            "camera": "",
            "lighting": "",
            "motion": "",
            "aesthetic": 0,
            "tags": [],
            "negative": "",
            "fps": float(meta["fps"] or 8),
            "frames": int(meta["frames"] or 8),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "bucket": bucket.name,
            "bucket_w": bucket.width,
            "bucket_h": bucket.height,
        }
        if flow_stats is not None:
            row.update(flow_stats.as_dict())
        rows.append(densify_row(row, fill_empty_labels=True))
    return write_manifest(rows, manifest_path)

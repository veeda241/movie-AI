"""Download linoyts Wan action datasets (MP4 + prompts) into Video Lab raw/.

These are small action packs (often ~10 clips each) generated with Wan 2.1 —
useful for niche motion training / LoRA-style concepts, not full Wan pretrain scale.

Videos provided by the Hugging Face dataset authors; credit dataset cards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from video_lab import DATA_ROOT, RAW_DIR, ensure_dirs

HF_WAN_INDEX_PATH = DATA_ROOT / "hf_wan_index.jsonl"

# Action packs shared by the user (linoyts/*)
DEFAULT_WAN_ACTION_DATASETS: tuple[str, ...] = (
    "linoyts/wan_putting_on_hat",
    "linoyts/wan_blowing_bubble_with_gum",
    "linoyts/wan_scrolling_on_phone",
    "linoyts/wan_shatter_effect",
    "linoyts/wan_shrugging",
    "linoyts/wan_shuffling_cards",
    "linoyts/wan_shaking_head",
    "linoyts/wan_buttoning_shirt",
    "linoyts/wan_blowing_out_candle",
    "linoyts/wan_blinking",
    "linoyts/wan_popping_balloon",
    "linoyts/wan_blockify_effect",
    "linoyts/wan_raising_eyebrows",
    "linoyts/wan_licking_lips",
    "linoyts/wan_crumble_disintegrate_effect",
    "linoyts/wan_saluting",
    "linoyts/wan_bouncing_ball",
    "linoyts/wan_putting_down_object",
    "linoyts/wan_pouring_liquid",
    "linoyts/wan_closing_umbrella",
    "linoyts/wan_showing_muscles",
    "linoyts/wan_doing_single_squat",
    "linoyts/wan_rolling_eyes",
    "linoyts/wan_clapping_hands",
    "linoyts/wan_peel_effect",
    "linoyts/wan_inflating_balloon",
    "linoyts/wan_curtseying",
    "linoyts/wan_facepalming",
    # Batch 2
    "linoyts/wan_brushing_teeth",
    "linoyts/wan_putting_on_glasses",
    "linoyts/wan_sketchify_effect",
    "linoyts/wan_deflate_effect",
    "linoyts/wan_overgrow_effect",
    "linoyts/wan_finger_counting",
    "linoyts/wan_catching_object",
    "linoyts/wan_eating",
    "linoyts/wan_inflate_effect",
    "linoyts/wan_crossing_arms",
    "linoyts/wan_petrify_effect",
    "linoyts/wan_picking_up_object",
    "linoyts/wan_burn_char_effect",
    "linoyts/wan_doing_robot_dance",
    "linoyts/wan_drinking_water",
    "linoyts/wan_pixelate_effect",
    # Batch 3
    "linoyts/wan_folding_paper",
    "linoyts/wan_doing_wave_arm",
    "linoyts/wan_doing_peace_sign",
    "linoyts/wan_doing_head_tilt",
    "linoyts/wan_jazz_hands",
    "linoyts/wan_emerge_effect",
    "linoyts/wan_origami_fold_effect",
    "linoyts/wan_jogging_in_place",
    "linoyts/wan_balancing_on_one_leg",
    "linoyts/wan_gasping",
    "linoyts/wan_doing_single_jumping_jack",
    "linoyts/wan_playing_piano",
    "linoyts/wan_puffing_cheeks",
    "linoyts/wan_frowning",
    "linoyts/wan_flipping_coin",
    "linoyts/wan_opening_book",
    "linoyts/wan_opening_closing_fan",
    "linoyts/wan_glitch_effect",
    "linoyts/wan_googly_eyes_effect",
)


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s or "clip")[:max_len]


def load_hf_wan_index(path: Path | None = None) -> dict[str, dict]:
    path = path or HF_WAN_INDEX_PATH
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = str(row.get("filename") or "")
            if name:
                out[name] = row
    return out


def append_hf_wan_index(rows: list[dict], path: Path | None = None) -> Path:
    path = path or HF_WAN_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def caption_from_hf_wan(filename: str, index: dict[str, dict] | None = None) -> str | None:
    index = index if index is not None else load_hf_wan_index()
    row = index.get(filename)
    if not row:
        return None
    return str(row.get("caption") or "").strip() or None


def _read_metadata(meta_path: Path) -> dict[str, str]:
    """Map file_name -> prompt from metadata.jsonl."""
    out: dict[str, str] = {}
    if not meta_path.exists():
        return out
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = str(row.get("file_name") or row.get("filename") or "")
            prompt = str(row.get("prompt") or row.get("caption") or "").strip()
            if name and prompt:
                out[name] = prompt
    return out


def download_wan_action_dataset(
    repo_id: str,
    *,
    out_dir: Path | None = None,
    token: str | None = None,
    log_fn=None,
) -> dict:
    """Download one linoyts/wan_* dataset into raw/."""

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    from huggingface_hub import hf_hub_download, list_repo_files

    ensure_dirs()
    out_dir = Path(out_dir or RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = token if token is not None else (os.environ.get("HF_TOKEN") or "").strip() or None

    action = repo_id.split("/")[-1].replace("wan_", "")
    action_slug = _slug(action)
    log(f"HF dataset: {repo_id}")

    files = list_repo_files(repo_id, repo_type="dataset", token=token)
    meta_local: Path | None = None
    if "metadata.jsonl" in files:
        meta_local = Path(
            hf_hub_download(repo_id, "metadata.jsonl", repo_type="dataset", token=token)
        )
    prompts = _read_metadata(meta_local) if meta_local else {}

    existing = load_hf_wan_index()
    new_rows: list[dict] = []
    downloaded = 0
    skipped = 0

    for name in files:
        if not name.lower().endswith((".mp4", ".webm", ".mov")):
            continue
        stem = Path(name).name
        caption = prompts.get(stem) or prompts.get(name) or f"{action.replace('_', ' ')}, realistic video"
        out_name = f"hf_wan_{action_slug}_{_slug(Path(stem).stem, 24)}.mp4"
        dest = out_dir / out_name
        if dest.exists() and dest.stat().st_size > 1000:
            skipped += 1
            continue
        src = Path(hf_hub_download(repo_id, name, repo_type="dataset", token=token))
        shutil.copy2(src, dest)
        row = {
            "filename": out_name,
            "path": str(dest.resolve()),
            "caption": caption,
            "dense_caption": caption,
            "source": "huggingface",
            "repo_id": repo_id,
            "action": action,
            "hf_file": name,
            "tags": ["wan_action", action_slug],
        }
        new_rows.append(row)
        downloaded += 1
        log(f"  + {out_name}")

    if new_rows:
        append_hf_wan_index(new_rows)
    return {
        "repo_id": repo_id,
        "downloaded": downloaded,
        "skipped": skipped,
        "out_dir": str(out_dir),
    }


def download_all_wan_action_datasets(
    repos: list[str] | tuple[str, ...] | None = None,
    *,
    out_dir: Path | None = None,
    log_fn=None,
) -> dict:
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    repos = tuple(repos or DEFAULT_WAN_ACTION_DATASETS)
    totals = {"downloaded": 0, "skipped": 0, "failed": 0, "repos": len(repos)}
    for repo in repos:
        try:
            summary = download_wan_action_dataset(repo, out_dir=out_dir, log_fn=log_fn)
            totals["downloaded"] += int(summary["downloaded"])
            totals["skipped"] += int(summary["skipped"])
        except Exception as e:
            totals["failed"] += 1
            log(f"FAIL {repo}: {e}")
    log(
        f"Done HF Wan actions: downloaded={totals['downloaded']} "
        f"skipped={totals['skipped']} failed={totals['failed']} "
        f"index={HF_WAN_INDEX_PATH}"
    )
    return totals

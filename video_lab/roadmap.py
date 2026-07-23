"""Live roadmap: what YOU add manually vs what the lab automates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from video_lab import DIT_CKPT_DIR, MANIFEST_PATH, RAW_DIR, SAMPLES_DIR, SMOKE_DIR, VAE_CKPT_DIR


@dataclass
class CheckItem:
    id: str
    phase: str
    who: str  # "you" | "lab"
    title: str
    how: str
    done: bool
    detail: str = ""


MANUAL_FEATURE_FIELDS = (
    ("caption", "Dense text description of what happens in the clip"),
    ("camera", "Camera move: pan / tilt / zoom / static / handheld / dolly"),
    ("lighting", "Lighting: neon / daylight / golden hour / low-key / …"),
    ("motion", "Subject motion: slow / fast / looping / particles / …"),
    ("aesthetic", "0–10 taste score (you decide what ‘good’ looks like)"),
    ("tags", "Free tags: genre, location, weather, style"),
    ("negative", "What this clip is NOT (helps later CFG / filters)"),
    ("dense_caption", "Auto/composed cinematography caption used for training"),
)


def _count_manifest(path: Path) -> tuple[int, int, dict[str, int]]:
    rows = 0
    real = 0
    fills = {k: 0 for k, _ in MANUAL_FEATURE_FIELDS}
    fills["flow"] = 0
    fills["bucket"] = 0
    if not path.exists():
        return 0, 0, fills
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = str(obj.get("path", ""))
        if p and p != "SYNTHETIC" and Path(p).exists():
            real += 1
        cap = str(obj.get("caption", "")).strip()
        if len(cap) >= 12:
            fills["caption"] += 1
        if str(obj.get("dense_caption", "")).strip():
            fills["dense_caption"] += 1
        if obj.get("flow_mean") is not None:
            fills["flow"] += 1
        if obj.get("bucket"):
            fills["bucket"] += 1
        for key in ("camera", "lighting", "motion", "tags", "negative", "aesthetic"):
            val = obj.get(key)
            if val is None or val == "" or val == []:
                continue
            if key == "aesthetic" and float(val) <= 0:
                continue
            fills[key] += 1
    return rows, real, fills


def _count_raw() -> int:
    if not RAW_DIR.exists():
        return 0
    return sum(1 for p in RAW_DIR.rglob("*") if p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"})


def _ckpt_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("config") or {}
    except Exception:
        return {}


def scan_checklist() -> list[CheckItem]:
    rows, real, fills = _count_manifest(MANIFEST_PATH)
    raw_n = _count_raw()
    smoke_n = sum(1 for _ in SMOKE_DIR.glob("*.mp4")) if SMOKE_DIR.exists() else 0
    vae_path = VAE_CKPT_DIR / "vae_last.pt"
    dit_path = DIT_CKPT_DIR / "dit_last.pt"
    vae_ok = vae_path.exists()
    dit_ok = dit_path.exists()
    vae_meta = _ckpt_meta(vae_path)
    dit_meta = _ckpt_meta(dit_path)
    samples = list(SAMPLES_DIR.glob("*.mp4")) if SAMPLES_DIR.exists() else []
    vae_compress_ok = int(vae_meta.get("spatial_compress", 0)) >= 8 and int(vae_meta.get("temporal_compress", 0)) >= 4
    patch = dit_meta.get("patch_size") or []
    patch_ok = isinstance(patch, (list, tuple)) and len(patch) == 3 and int(patch[1]) >= 2
    stage = str(dit_meta.get("stage") or vae_meta.get("stage") or "")

    items: list[CheckItem] = [
        CheckItem(
            "raw_clips",
            "Phase A — Data",
            "you",
            "Add real video clips",
            f"Copy MP4/WebM/MOV into `{RAW_DIR}` (aim 50+, then 500+, then 5k+).",
            raw_n >= 1,
            f"{raw_n} file(s) in raw/",
        ),
        CheckItem(
            "smoke_ok",
            "Phase A — Data",
            "lab",
            "Smoke dataset available",
            "Gradio Data → Create smoke dataset (toy motion patterns).",
            smoke_n >= 4,
            f"{smoke_n} smoke clip(s)",
        ),
        CheckItem(
            "manifest",
            "Phase A — Data",
            "lab",
            "Manifest built",
            "Data → Curate raw → manifest (or smoke).",
            rows >= 1,
            f"{rows} row(s), {real} with existing video path",
        ),
        CheckItem(
            "flow",
            "Phase A — Data",
            "lab",
            "Optical-flow scores on rows",
            "Curate with optical-flow filter enabled (Farneback mean/var).",
            fills["flow"] >= max(1, rows // 2) and rows > 0,
            f"{fills['flow']}/{rows} have flow_mean",
        ),
        CheckItem(
            "dense_caption",
            "Phase A — Data",
            "lab",
            "Dense captions written",
            "Data → Recaption manifest (or Labels save).",
            fills["dense_caption"] >= max(1, rows // 2) and rows > 0,
            f"{fills['dense_caption']}/{rows} dense_caption",
        ),
        CheckItem(
            "bucket",
            "Phase A — Data",
            "lab",
            "Aspect buckets assigned",
            "Curate assigns bucket; Train selects one bucket at a time.",
            fills["bucket"] >= max(1, rows // 2) and rows > 0,
            f"{fills['bucket']}/{rows} have bucket",
        ),
        CheckItem(
            "captions",
            "Phase A — Data",
            "you",
            "Write base captions",
            "Labels tab: clear `caption` (who/what/setting).",
            fills["caption"] >= max(1, rows // 2) and rows > 0,
            f"{fills['caption']}/{rows} captions filled",
        ),
        CheckItem(
            "camera",
            "Phase A — Data",
            "you",
            "Add camera labels",
            'Labels: `"camera": "slow pan"` etc.',
            fills["camera"] >= 1,
            f"{fills['camera']}/{rows} have camera",
        ),
        CheckItem(
            "lighting",
            "Phase A — Data",
            "you",
            "Add lighting labels",
            'Labels: `"lighting": "neon night"` / daylight / golden hour.',
            fills["lighting"] >= 1,
            f"{fills['lighting']}/{rows} have lighting",
        ),
        CheckItem(
            "motion",
            "Phase A — Data",
            "you",
            "Add motion labels",
            'Labels: `"motion": "particles drifting"` etc.',
            fills["motion"] >= 1,
            f"{fills['motion']}/{rows} have motion",
        ),
        CheckItem(
            "aesthetic",
            "Phase A — Data",
            "you",
            "Score aesthetics",
            'Labels: `"aesthetic": 7` (0–10).',
            fills["aesthetic"] >= 1,
            f"{fills['aesthetic']}/{rows} scored",
        ),
        CheckItem(
            "vae",
            "Phase B — Train",
            "lab",
            "Train causal VAE",
            "Train tab → Train VAE (8× spatial / 4× temporal).",
            vae_ok,
            "vae_last.pt" if vae_ok else "missing",
        ),
        CheckItem(
            "vae_compress",
            "Phase B — Architecture",
            "lab",
            "VAE compression 8×S / 4×T",
            "Retrain VAE after architecture upgrade so checkpoint meta matches.",
            vae_compress_ok,
            f"spatial={vae_meta.get('spatial_compress')} temporal={vae_meta.get('temporal_compress')}",
        ),
        CheckItem(
            "dit",
            "Phase B — Train",
            "lab",
            "Train spacetime-patch DiT",
            "Train tab → Train DiT after VAE.",
            dit_ok,
            "dit_last.pt" if dit_ok else "missing",
        ),
        CheckItem(
            "dit_patch",
            "Phase B — Architecture",
            "lab",
            "Spacetime patches enabled",
            "DiT checkpoint stores patch_size (e.g. 1×2×2).",
            patch_ok,
            f"patch_size={patch}",
        ),
        CheckItem(
            "stage",
            "Phase B — Curriculum",
            "lab",
            "Progressive stage recorded",
            "Train with Stage 1/2/3 preset; stage saved in checkpoint meta.",
            bool(stage),
            f"stage={stage or 'none'}",
        ),
        CheckItem(
            "sample",
            "Phase B — Generate",
            "lab",
            "Generate a sample MP4",
            "Generate tab → prompt close to your captions.",
            len(samples) >= 1,
            f"{len(samples)} sample(s)",
        ),
        CheckItem(
            "scale_data",
            "Phase C — Compete (later)",
            "you",
            "Scale dataset",
            "Grow beyond smoke: 1k → 10k → 100k+ real clips.",
            real >= 50,
            f"{real} real clips (target 50+)",
        ),
        CheckItem(
            "compute",
            "Phase C — Compete (later)",
            "you",
            "More GPU / cloud",
            "RTX 3050 is fine for research; Veo-class needs multi-GPU weeks.",
            False,
            "Manual decision when Phase B looks stable",
        ),
    ]
    return items


def checklist_markdown(items: list[CheckItem] | None = None) -> str:
    items = items or scan_checklist()
    lines = [
        "# Own-model checklist",
        "",
        "Items marked **YOU** are manual. **LAB** is done in Gradio.",
        "Phase A = data density; Phase B = stronger VAE/DiT; Phase C deferred (DPO / distill / open SOTA).",
        "",
    ]
    phase = None
    for it in items:
        if it.phase != phase:
            phase = it.phase
            lines.append(f"## {phase}")
            lines.append("")
        mark = "✅" if it.done else "⬜"
        who = "YOU" if it.who == "you" else "LAB"
        lines.append(f"- {mark} **[{who}] {it.title}** — {it.detail}")
        lines.append(f"  - {it.how}")
        lines.append("")
    lines.append("## Manifest fields")
    lines.append("")
    for key, desc in MANUAL_FEATURE_FIELDS:
        lines.append(f"- `{key}` — {desc}")
    lines.append("")
    lines.append("Also written by curation: `flow_mean`, `flow_var`, `bucket`, `width`, `height`.")
    return "\n".join(lines)


def write_manifest_template(path: Path | None = None) -> Path:
    path = path or (MANIFEST_PATH.parent / "manifest_template_example.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    example = {
        "path": str((RAW_DIR / "YOUR_CLIP.mp4").resolve()),
        "caption": "REPLACE: dense description of subject, action, setting",
        "dense_caption": "REPLACE after Recaption button",
        "camera": "REPLACE: static | pan left | dolly forward",
        "lighting": "REPLACE: daylight | neon | golden hour",
        "motion": "REPLACE: what moves",
        "aesthetic": 5,
        "tags": ["REPLACE_TAG"],
        "negative": "blurry, watermark, text overlay",
        "bucket": "square_128",
        "fps": 24,
        "frames": 48,
    }
    path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

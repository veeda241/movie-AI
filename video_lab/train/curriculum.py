"""Progressive multi-stage training presets (lab-scale)."""

from __future__ import annotations

from dataclasses import dataclass

from video_lab.data.buckets import ASPECT_BUCKETS, AspectBucket, parse_bucket_choice


@dataclass(frozen=True)
class StagePreset:
    name: str
    label: str
    frames: int
    bucket: str
    min_aesthetic: float
    description: str


STAGES: tuple[StagePreset, ...] = (
    StagePreset(
        "stage1",
        "Stage 1 — Spatial",
        frames=8,
        bucket="square_96",
        min_aesthetic=0.0,
        description="Short clips / low res — learn layout and texture.",
    ),
    StagePreset(
        "stage2",
        "Stage 2 — Low-res motion",
        frames=16,
        bucket="square_128",
        min_aesthetic=0.0,
        description="Default motion pretrain at 128².",
    ),
    StagePreset(
        "stage3",
        "Stage 3 — Higher-res curated",
        frames=24,
        bucket="square_256",
        min_aesthetic=6.0,
        description="Higher res + aesthetic filter (needs VRAM).",
    ),
    StagePreset(
        "niche_laptop",
        "Niche laptop — 256² / 8f",
        frames=8,
        bucket="square_256",
        min_aesthetic=0.0,
        description="6GB attempt at 256² (short clips).",
    ),
    StagePreset(
        "niche_24gb",
        "Niche 24GB — 256² / 24f",
        frames=24,
        bucket="square_256",
        min_aesthetic=6.0,
        description="Target profile for 4090/A5000 + 1k–10k clips.",
    ),
)

STAGE_BY_NAME = {s.name: s for s in STAGES}


def list_stage_labels() -> list[str]:
    return [s.label for s in STAGES]


def resolve_stage(choice: str | None) -> StagePreset:
    if not choice:
        return STAGE_BY_NAME["stage2"]
    c = str(choice).strip().lower()
    for s in STAGES:
        if c == s.name or c.startswith(s.name) or choice == s.label:
            return s
    return STAGE_BY_NAME["stage2"]


def stage_bucket(stage: StagePreset) -> AspectBucket:
    return parse_bucket_choice(stage.bucket)


def apply_stage_to_train_kwargs(stage_choice: str | None, **overrides) -> dict:
    stage = resolve_stage(stage_choice)
    bucket = stage_bucket(stage)
    kwargs = {
        "frames": stage.frames,
        "height": bucket.height,
        "width": bucket.width,
        "bucket": bucket.name,
        "min_aesthetic": stage.min_aesthetic,
        "train_stage": stage.name,
    }
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs

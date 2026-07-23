"""Niche-model profiles: 256–512p targets for 24GB+ GPUs (plus laptop-safe variant)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NicheProfile:
    name: str
    label: str
    frames: int
    height: int
    width: int
    bucket: str
    dit_size: str
    vae_steps: int
    dit_steps: int
    min_aesthetic: float
    min_vram_gb: float
    description: str


PROFILES: tuple[NicheProfile, ...] = (
    NicheProfile(
        name="niche_laptop",
        label="Niche laptop (3050 6GB) — 256² / 8f",
        frames=8,
        height=256,
        width=256,
        bucket="square_256",
        dit_size="small",
        vae_steps=300,
        dit_steps=400,
        min_aesthetic=0.0,
        min_vram_gb=5.0,
        description="Aggressive VRAM-safe niche attempt. Expect slow train; may still OOM.",
    ),
    NicheProfile(
        name="niche_24gb",
        label="Niche 24GB — 256² / 24f",
        frames=24,
        height=256,
        width=256,
        bucket="square_256",
        dit_size="medium",
        vae_steps=2000,
        dit_steps=4000,
        min_aesthetic=6.0,
        min_vram_gb=20.0,
        description="Decent niche target on 4090/A5000: days on 1k–10k clips.",
    ),
    NicheProfile(
        name="niche_24gb_512",
        label="Niche 24GB — 512² / 16f",
        frames=16,
        height=512,
        width=512,
        bucket="square_256",  # letterbox/pad path still uses H/W overrides
        dit_size="medium",
        vae_steps=1500,
        dit_steps=3000,
        min_aesthetic=7.0,
        min_vram_gb=22.0,
        description="Higher-res niche; prefer 24GB+ with AMP. Override bucket dims via H/W.",
    ),
)

PROFILE_BY_NAME = {p.name: p for p in PROFILES}


def list_niche_labels() -> list[str]:
    return [p.label for p in PROFILES]


def resolve_niche(choice: str | None) -> NicheProfile:
    if not choice:
        return PROFILE_BY_NAME["niche_laptop"]
    c = str(choice).strip().lower()
    for p in PROFILES:
        if c == p.name or c.startswith(p.name) or choice == p.label:
            return p
    return PROFILE_BY_NAME["niche_laptop"]


def niche_train_kwargs(choice: str | None) -> dict:
    p = resolve_niche(choice)
    return {
        "frames": p.frames,
        "height": p.height,
        "width": p.width,
        "bucket": p.bucket,
        "dit_size": p.dit_size,
        "vae_steps": p.vae_steps,
        "dit_steps": p.dit_steps,
        "min_aesthetic": p.min_aesthetic,
        "train_stage": f"niche:{p.name}",
        "profile": p,
    }

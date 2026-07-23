"""Video Model Lab — research MVP + fine-tune toolkit."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "video_lab"
OUTPUT_ROOT = ROOT / "outputs" / "video_lab"

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SMOKE_DIR = DATA_ROOT / "smoke"
MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"

VAE_CKPT_DIR = OUTPUT_ROOT / "vae"
DIT_CKPT_DIR = OUTPUT_ROOT / "dit"
LORA_CKPT_DIR = OUTPUT_ROOT / "lora"
SAMPLES_DIR = OUTPUT_ROOT / "samples"


def ensure_dirs() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        SMOKE_DIR,
        VAE_CKPT_DIR,
        DIT_CKPT_DIR,
        LORA_CKPT_DIR,
        SAMPLES_DIR,
        DATA_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

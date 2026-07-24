from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_lab import DIT_CKPT_DIR, LORA_CKPT_DIR, MANIFEST_PATH, SAMPLES_DIR, VAE_CKPT_DIR


@dataclass
class LabConfig:
    manifest_path: Path = MANIFEST_PATH
    vae_dir: Path = VAE_CKPT_DIR
    dit_dir: Path = DIT_CKPT_DIR
    lora_dir: Path = LORA_CKPT_DIR
    samples_dir: Path = SAMPLES_DIR

    # Defaults aligned with niche_laptop train (match Generate to Train)
    frames: int = 8
    fps: int = 12
    duration_sec: float = 0.7
    height: int = 256
    width: int = 256
    vae_latent_channels: int = 4
    dit_hidden: int = 192
    dit_layers: int = 4
    dit_heads: int = 4
    # "small" | "medium" — medium needs more VRAM / longer train
    dit_size: str = "small"
    batch_size: int = 1
    lr: float = 1e-4
    vae_lr: float = 1e-4
    dit_lr: float = 1e-4
    vae_steps: int = 600
    dit_steps: int = 800
    sample_steps: int = 24
    cfg_scale: float = 2.5
    min_aesthetic: float = 0.0

    # Data quality (Phase A)
    min_flow: float = 0.15
    max_flow: float = 12.0
    max_flow_var: float = 40.0
    default_bucket: str = "square_256"

    # VAE compression (Phase B): spatial_stride product / temporal
    vae_spatial_compress: int = 8
    vae_temporal_compress: int = 4
    vae_base_channels: int = 48

    # DiT spacetime patch (Phase B): (t, h, w)
    dit_patch_t: int = 1
    dit_patch_h: int = 2
    dit_patch_w: int = 2
    train_stage: str = "stage2"  # stage1 | stage2 | stage3

    # CogVideoX + LoRA (pre-trained model fine-tune)
    base_t2v_model: str = "THUDM/CogVideoX-2b"
    lora_rank: int = 16
    lora_steps: int = 100

    def dit_dims(self) -> tuple[int, int, int]:
        if str(self.dit_size).lower() == "medium":
            return 256, 6, 4
        return self.dit_hidden, self.dit_layers, self.dit_heads

    def dit_patch_size(self) -> tuple[int, int, int]:
        return (int(self.dit_patch_t), int(self.dit_patch_h), int(self.dit_patch_w))

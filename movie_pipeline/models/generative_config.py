from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ModelFamily(StrEnum):
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"
    FLUX = "flux"
    ANIMATEDIFF = "animatediff"
    COGVIDEOX = "cogvideox"
    MOTIF_VIDEO = "motif-video"


@dataclass(frozen=True)
class ImageGenerationConfig:
    family: ModelFamily = ModelFamily.STABLE_DIFFUSION_XL
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id: str | None = "madebyollin/sdxl-vae-fp16-fix"
    lora_path: str | None = None
    scheduler: str = "dpm-solver++"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 6.5
    negative_prompt: str = (
        "low quality, blurry, distorted anatomy, unreadable text, watermark, logo"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "output" / "images"
    )

    @classmethod
    def from_env(cls, output_dir: Path | None = None) -> "ImageGenerationConfig":
        default_out = Path(__file__).resolve().parents[1] / "output" / "images"
        return cls(
            family=ModelFamily(os.environ.get("IMAGE_MODEL_FAMILY", cls.family.value)),
            model_id=os.environ.get("IMAGE_MODEL_ID", cls.model_id),
            vae_id=os.environ.get("IMAGE_VAE_ID", cls.vae_id or "") or None,
            lora_path=os.environ.get("IMAGE_LORA_PATH") or None,
            scheduler=os.environ.get("IMAGE_SCHEDULER", cls.scheduler),
            width=int(os.environ.get("IMAGE_WIDTH", str(cls.width))),
            height=int(os.environ.get("IMAGE_HEIGHT", str(cls.height))),
            num_inference_steps=int(os.environ.get("IMAGE_STEPS", str(cls.num_inference_steps))),
            guidance_scale=float(os.environ.get("IMAGE_GUIDANCE", str(cls.guidance_scale))),
            negative_prompt=os.environ.get("IMAGE_NEGATIVE_PROMPT", cls.negative_prompt),
            output_dir=Path(output_dir) if output_dir is not None else Path(os.environ.get("IMAGE_OUTPUT_DIR", str(default_out))),
        )


@dataclass(frozen=True)
class VideoGenerationConfig:
    family: ModelFamily = ModelFamily.MOTIF_VIDEO
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B"
    provider: str = "fal-ai"
    num_frames: int = 8
    num_inference_steps: int = 10
    fps: int = 24
    guidance_scale: float = 6.0
    temporal_strategy: str = "video-dit"
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "output"
    )

    @classmethod
    def from_env(cls) -> "VideoGenerationConfig":
        return cls(
            family=ModelFamily(os.environ.get("VIDEO_MODEL_FAMILY", cls.family.value)),
            model_id=os.environ.get("HF_VIDEO_MODEL", cls.model_id),
            provider=os.environ.get("HF_VIDEO_PROVIDER", cls.provider),
            num_frames=int(os.environ.get("HF_VIDEO_REMOTE_FRAMES", str(cls.num_frames))),
            num_inference_steps=int(
                os.environ.get("HF_VIDEO_REMOTE_STEPS", str(cls.num_inference_steps))
            ),
            fps=int(os.environ.get("HF_LOCAL_VIDEO_FPS", str(cls.fps))),
            guidance_scale=float(os.environ.get("VIDEO_GUIDANCE", str(cls.guidance_scale))),
            temporal_strategy=os.environ.get("VIDEO_TEMPORAL_STRATEGY", cls.temporal_strategy),
        )


@dataclass(frozen=True)
class FineTuneConfig:
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    dataset_dir: Path = Path("data/images")
    output_dir: Path = Path("movie_pipeline/output/lora")
    resolution: int = 1024
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1200
    rank: int = 16
    validation_prompt: str = "cinematic production still, dramatic lighting"

    def to_accelerate_args(self) -> list[str]:
        return [
            f"--pretrained_model_name_or_path={self.base_model_id}",
            f"--train_data_dir={self.dataset_dir}",
            f"--output_dir={self.output_dir}",
            f"--resolution={self.resolution}",
            f"--train_batch_size={self.train_batch_size}",
            f"--gradient_accumulation_steps={self.gradient_accumulation_steps}",
            f"--learning_rate={self.learning_rate}",
            f"--max_train_steps={self.max_train_steps}",
            f"--rank={self.rank}",
            f"--validation_prompt={self.validation_prompt}",
        ]

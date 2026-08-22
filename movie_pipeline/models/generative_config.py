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
    MINIMAX_H3 = "minimax-h3"


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
class MiniMaxH3Config:
    """Configuration for MiniMax H3 video generation via the MiniMax platform API."""

    model_name: str = "MiniMax-H3"
    base_url: str = "https://api.minimax.io"
    duration: int = 5
    resolution: str = "768P"
    ratio: str = "16:9"
    poll_interval: int = 10
    max_poll_attempts: int = 60
    request_timeout: int = 30
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "output"
    )

    @classmethod
    def from_env(cls) -> "MiniMaxH3Config":
        return cls(
            base_url=os.environ.get("MINIMAX_API_BASE", cls.base_url),
            duration=int(os.environ.get("MINIMAX_DURATION", str(cls.duration))),
            resolution=os.environ.get("MINIMAX_RESOLUTION", cls.resolution),
            ratio=os.environ.get("MINIMAX_RATIO", cls.ratio),
            poll_interval=int(os.environ.get("MINIMAX_POLL_INTERVAL", str(cls.poll_interval))),
            max_poll_attempts=int(os.environ.get("MINIMAX_MAX_POLL_ATTEMPTS", str(cls.max_poll_attempts))),
            request_timeout=int(os.environ.get("MINIMAX_REQUEST_TIMEOUT", str(cls.request_timeout))),
        )


@dataclass(frozen=True)
class VideoGenerationConfig:
    family: ModelFamily = ModelFamily.MOTIF_VIDEO
    # Defaults match .env.example (free HF Inference tier). UI "wan-2.2" overrides to fal.
    model_id: str = "ali-vilab/text-to-video-ms-1.7b"
    provider: str = "hf-inference"
    num_frames: int = 16
    num_inference_steps: int = 20
    fps: int = 24
    guidance_scale: float = 6.0
    temporal_strategy: str = "video-dit"
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "output"
    )

    @classmethod
    def from_env(cls) -> "VideoGenerationConfig":
        minimax_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if minimax_key:
            default_family = ModelFamily.MINIMAX_H3
            default_model = "MiniMax-H3"
            default_provider = "minimax"
        else:
            default_family = ModelFamily.MOTIF_VIDEO
            default_model = "ali-vilab/text-to-video-ms-1.7b"
            default_provider = "hf-inference"
        return cls(
            family=ModelFamily(os.environ.get("VIDEO_MODEL_FAMILY", default_family.value)),
            model_id=os.environ.get("HF_VIDEO_MODEL", default_model),
            provider=os.environ.get("HF_VIDEO_PROVIDER", default_provider),
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

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from movie_pipeline.models.generative_config import ImageGenerationConfig


class ImageGenerationClient:
    def __init__(self, config: ImageGenerationConfig | None = None) -> None:
        self.config = config or ImageGenerationConfig.from_env()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Any | None = None

    def generate(self, prompt: str, scene_number: int, seed: int | None = None) -> str:
        try:
            return self._generate_with_diffusers(prompt, scene_number, seed)
        except ImportError as exc:
            raise RuntimeError(
                "Image generation needs optional model dependencies. Install them with "
                "`pip install -r requirements-model.txt`."
            ) from exc

    def _generate_with_diffusers(self, prompt: str, scene_number: int, seed: int | None) -> str:
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        if self._pipeline is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            vae = None
            if self.config.vae_id:
                vae = AutoencoderKL.from_pretrained(self.config.vae_id, torch_dtype=dtype)

            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                vae=vae,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            if self.config.scheduler.lower() in {"dpm-solver++", "dpmpp", "dpm"}:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                )
            if self.config.lora_path:
                pipe.load_lora_weights(self.config.lora_path)

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                pipe.enable_attention_slicing()
            else:
                pipe = pipe.to("cpu")

            self._pipeline = pipe

        generator = None
        if seed is not None:
            device = "cuda" if self._pipeline.device.type == "cuda" else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)

        image = self._pipeline(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]

        output_path = self.config.output_dir / f"scene_{scene_number}_keyframe.png"
        image.save(output_path)
        return str(output_path)


def seed_from_prompt(prompt: str, scene_number: int) -> int:
    digest = hashlib.sha256(f"{scene_number}:{prompt}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)

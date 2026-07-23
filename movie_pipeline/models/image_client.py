from __future__ import annotations

import hashlib
import math
import os
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

from movie_pipeline.models.generative_config import ImageGenerationConfig


class ImageGenerationClient:
    def __init__(
        self,
        config: ImageGenerationConfig | None = None,
        *,
        output_dir: Path | None = None,
    ) -> None:
        base = config or ImageGenerationConfig.from_env()
        if output_dir is not None:
            base = replace(base, output_dir=Path(output_dir))
        self.config = base
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Any | None = None

    def generate(
        self,
        prompt: str,
        scene_number: int = 0,
        seed: int | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> str:
        target = (
            Path(output_path)
            if output_path is not None
            else self.config.output_dir / f"scene_{scene_number}_keyframe.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_seed = seed if seed is not None else seed_from_prompt(prompt, scene_number)

        try:
            return self._generate_with_diffusers(prompt, target, resolved_seed)
        except ImportError:
            return self._generate_local_still(prompt, scene_number, target, resolved_seed)
        except Exception as exc:
            print(f"[ImageGenerationClient] diffusers failed ({exc}); using local still.", flush=True)
            return self._generate_local_still(prompt, scene_number, target, resolved_seed)

    def generate_still(
        self,
        prompt: str,
        *,
        output_path: str | Path,
        seed: int | None = None,
    ) -> str:
        return self.generate(prompt, scene_number=0, seed=seed, output_path=output_path)

    def _generate_with_diffusers(self, prompt: str, output_path: Path, seed: int) -> str:
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

        image.save(output_path)
        return str(output_path)

    def _generate_local_still(self, prompt: str, scene_number: int, output_path: Path, seed: int) -> str:
        from PIL import Image, ImageDraw, ImageFont

        width = self.config.width if self.config.width % 2 == 0 else self.config.width - 1
        height = self.config.height if self.config.height % 2 == 0 else self.config.height - 1
        width = min(max(width, 512), 1280)
        height = min(max(height, 512), 1280)

        palettes = [
            ((12, 14, 22), (48, 36, 28), (196, 140, 72), (240, 228, 210)),
            ((8, 16, 24), (24, 56, 72), (88, 168, 176), (220, 236, 240)),
            ((18, 12, 20), (72, 28, 48), (200, 96, 88), (245, 220, 210)),
        ]
        top, mid, accent, highlight = palettes[seed % len(palettes)]
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            t = y / max(height - 1, 1)
            color = tuple(int(top[i] + (mid[i] - top[i]) * t) for i in range(3))
            draw.line((0, y, width, y), fill=color)

        cx, cy = int(width * 0.62), int(height * 0.38)
        radius = int(min(width, height) * 0.28)
        for i in range(radius, 0, -8):
            alpha = int(40 + 80 * (1 - i / radius))
            glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            gdraw = ImageDraw.Draw(glow)
            gdraw.ellipse(
                (cx - i, cy - i, cx + i, cy + i),
                fill=(accent[0], accent[1], accent[2], alpha),
            )
            image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
            draw = ImageDraw.Draw(image)

        for n in range(18):
            angle = (seed % 97) / 97 + n * 0.35
            px = int(width * (0.15 + 0.7 * ((math.sin(angle * 3 + seed) + 1) / 2)))
            py = int(height * (0.2 + 0.55 * ((math.cos(angle * 2 + n) + 1) / 2)))
            size = 3 + (n + seed) % 7
            draw.ellipse((px, py, px + size, py + size), fill=highlight)

        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
        try:
            title_font = ImageFont.truetype(str(font_path), 28)
            body_font = ImageFont.truetype(str(font_path), 18)
        except Exception:
            title_font = ImageFont.load_default()
            body_font = title_font

        panel = (36, height - 160, width - 36, height - 36)
        draw.rounded_rectangle(panel, radius=18, fill=(8, 10, 14))
        draw.text((52, height - 140), f"Still {scene_number}" if scene_number else "Movie Flow", font=title_font, fill=highlight)
        excerpt = textwrap.fill(textwrap.shorten(prompt, width=140, placeholder="..."), width=52)
        draw.text((52, height - 100), excerpt, font=body_font, fill=(220, 214, 200))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return str(output_path)


def seed_from_prompt(prompt: str, scene_number: int) -> int:
    digest = hashlib.sha256(f"{scene_number}:{prompt}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)

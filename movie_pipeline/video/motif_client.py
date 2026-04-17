from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import textwrap
from pathlib import Path
from typing import Any, cast

from huggingface_hub import InferenceClient


class MotifClient:
    REMOTE_PROVIDER = os.environ.get("HF_VIDEO_PROVIDER", "fal-ai")
    REMOTE_MODEL = os.environ.get("HF_VIDEO_MODEL", "Wan-AI/Wan2.2-T2V-A14B")
    REMOTE_NUM_FRAMES = int(os.environ.get("HF_VIDEO_REMOTE_FRAMES", "8"))
    REMOTE_NUM_INFERENCE_STEPS = int(os.environ.get("HF_VIDEO_REMOTE_STEPS", "10"))
    REMOTE_TIMEOUT_SECONDS = int(os.environ.get("HF_VIDEO_REMOTE_TIMEOUT", "180"))

    LOCAL_WIDTH = int(os.environ.get("HF_LOCAL_VIDEO_WIDTH", "960"))
    LOCAL_HEIGHT = int(os.environ.get("HF_LOCAL_VIDEO_HEIGHT", "540"))
    LOCAL_FPS = int(os.environ.get("HF_LOCAL_VIDEO_FPS", "24"))
    LOCAL_DURATION_SECONDS = int(os.environ.get("HF_LOCAL_VIDEO_SECONDS", "4"))

    def __init__(self) -> None:
        self.token = os.environ.get("HF_TOKEN", "")
        self.output_dir = Path(__file__).resolve().parents[1] / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, video_prompt: str, scene_number: int) -> str:
        if self.token.strip():
            try:
                remote_video = self._generate_remote_video(video_prompt, scene_number)
                if remote_video:
                    return self._write_video_file(scene_number, remote_video)
            except Exception as exc:
                print(f"[MotifClient] scene {scene_number} remote video failed: {exc}", flush=True)
        else:
            print(f"[MotifClient] scene {scene_number} missing HF_TOKEN, using local fallback.", flush=True)

        try:
            return self._generate_local_video(video_prompt, scene_number)
        except Exception as exc:
            print(f"[MotifClient] scene {scene_number} local fallback failed: {exc}", flush=True)
            return ""

    def _generate_remote_video(self, video_prompt: str, scene_number: int) -> bytes:
        client = InferenceClient(
            provider=cast(Any, self.REMOTE_PROVIDER),
            api_key=self.token.strip(),
            timeout=self.REMOTE_TIMEOUT_SECONDS,
        )
        seed = self._seed_from_text(video_prompt, scene_number)
        video = client.text_to_video(
            video_prompt,
            model=self.REMOTE_MODEL,
            num_frames=self.REMOTE_NUM_FRAMES,
            num_inference_steps=self.REMOTE_NUM_INFERENCE_STEPS,
            seed=seed,
        )

        if isinstance(video, bytes):
            return video
        if isinstance(video, bytearray):
            return bytes(video)

        raise RuntimeError(f"Unexpected remote video response type: {type(video).__name__}")

    def _write_video_file(self, scene_number: int, video_bytes: bytes) -> str:
        out_path = self.output_dir / f"scene_{scene_number}_video.mp4"
        out_path.write_bytes(video_bytes)
        return str(out_path)

    def _generate_local_video(self, video_prompt: str, scene_number: int) -> str:
        import imageio_ffmpeg

        seed = self._seed_from_text(video_prompt, scene_number)
        palette = self._build_palette(seed)
        background = self._build_background(self.LOCAL_WIDTH + 140, self.LOCAL_HEIGHT + 140, palette)
        frame_count = self.LOCAL_FPS * self.LOCAL_DURATION_SECONDS
        font_title = self._load_font(28)
        font_body = self._load_font(18)
        out_path = self.output_dir / f"scene_{scene_number}_video.mp4"

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.LOCAL_WIDTH}x{self.LOCAL_HEIGHT}",
            "-r",
            str(self.LOCAL_FPS),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is not available")

            with process.stdin:
                for index in range(frame_count):
                    progress = index / max(frame_count - 1, 1)
                    frame = self._render_local_frame(
                        background=background,
                        prompt=video_prompt,
                        scene_number=scene_number,
                        palette=palette,
                        progress=progress,
                        seed=seed,
                        font_title=font_title,
                        font_body=font_body,
                    ).convert("RGB")
                    process.stdin.write(frame.tobytes())

            stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {return_code}: {stderr_output.strip() or 'no stderr output'}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

        return str(out_path)

    def _render_local_frame(
        self,
        *,
        background: Any,
        prompt: str,
        scene_number: int,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        progress: float,
        seed: int,
        font_title: Any,
        font_body: Any,
    ) -> Any:
        from PIL import Image, ImageDraw

        width, height = self.LOCAL_WIDTH, self.LOCAL_HEIGHT
        sway_x = int((math.sin(progress * math.tau + (seed % 997) / 97.0) + 1.0) * 0.5 * 70)
        sway_y = int((math.cos(progress * math.tau * 0.85 + (seed % 577) / 83.0) + 1.0) * 0.5 * 70)
        frame = background.crop((sway_x, sway_y, sway_x + width, sway_y + height)).convert("RGBA")

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        self._draw_particles(draw, width, height, prompt, progress, seed, palette)
        self._draw_title_panel(draw, width, height, prompt, scene_number, font_title, font_body)

        frame = Image.alpha_composite(frame, overlay)
        return frame.convert("RGB")

    def _draw_particles(
        self,
        draw: Any,
        width: int,
        height: int,
        prompt: str,
        progress: float,
        seed: int,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> None:
        prompt_lower = prompt.lower()
        weather_keywords = ["rain", "storm", "snow", "wind", "fog", "night", "street", "city"]
        density = 36 + 8 * sum(keyword in prompt_lower for keyword in weather_keywords)
        palette_glow = palette[2]
        sparkle = palette[3]
        particle_rng = random.Random(seed + int(progress * 1000))

        for _ in range(density):
            x = particle_rng.randint(-width // 5, width + width // 5)
            y = particle_rng.randint(-height // 5, height + height // 5)
            streak_length = particle_rng.randint(12, 42)
            drift = particle_rng.randint(-6, 6)
            alpha = particle_rng.randint(50, 140)
            if "rain" in prompt_lower or "storm" in prompt_lower or "street" in prompt_lower:
                color = (sparkle[0], sparkle[1], sparkle[2], alpha)
                draw.line((x, y, x + drift, y + streak_length), fill=color, width=1)
            else:
                size = particle_rng.randint(2, 5)
                draw.ellipse((x, y, x + size, y + size), fill=(sparkle[0], sparkle[1], sparkle[2], alpha))

        for band_index in range(3):
            wave = math.sin(progress * math.tau * (0.75 + band_index * 0.18) + band_index)
            x_center = int((0.18 + band_index * 0.28 + 0.08 * wave) * width)
            y_center = int((0.22 + band_index * 0.15) * height)
            radius = int(min(width, height) * (0.14 + band_index * 0.04))
            alpha = 26 + band_index * 12
            draw.ellipse(
                (x_center - radius, y_center - radius, x_center + radius, y_center + radius),
                fill=(palette_glow[0], palette_glow[1], palette_glow[2], alpha),
            )

    def _draw_title_panel(
        self,
        draw: Any,
        width: int,
        height: int,
        prompt: str,
        scene_number: int,
        font_title: Any,
        font_body: Any,
    ) -> None:
        panel_margin = 34
        panel_top = height - 170
        panel_left = panel_margin
        panel_right = width - panel_margin
        panel_bottom = height - panel_margin

        draw.rounded_rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            radius=24,
            fill=(6, 10, 18, 190),
            outline=(255, 255, 255, 28),
            width=1,
        )

        draw.text((panel_left + 22, panel_top + 18), f"Scene {scene_number}", font=font_title, fill=(255, 255, 255, 240))
        prompt_excerpt = textwrap.fill(textwrap.shorten(prompt, width=170, placeholder="..."), width=56)
        draw.text(
            (panel_left + 22, panel_top + 64),
            prompt_excerpt,
            font=font_body,
            fill=(238, 244, 255, 220),
            spacing=4,
        )

        footer = "Local cinematic fallback"
        draw.text((panel_left + 22, panel_bottom - 30), footer, font=font_body, fill=(184, 204, 255, 170))

    def _build_background(
        self,
        width: int,
        height: int,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> Any:
        from PIL import Image, ImageDraw

        top, middle, accent, highlight = palette
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)

        for y in range(height):
            ratio = y / max(height - 1, 1)
            if ratio < 0.5:
                color = self._blend(top, middle, ratio * 2.0)
            else:
                color = self._blend(middle, accent, (ratio - 0.5) * 2.0)
            draw.line((0, y, width, y), fill=color + (255,))

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        self._draw_glow(overlay_draw, width, height, (int(width * 0.26), int(height * 0.34)), int(min(width, height) * 0.44), highlight, 72)
        self._draw_glow(overlay_draw, width, height, (int(width * 0.76), int(height * 0.24)), int(min(width, height) * 0.26), accent, 58)
        self._draw_glow(overlay_draw, width, height, (int(width * 0.54), int(height * 0.72)), int(min(width, height) * 0.34), middle, 42)

        for inset in range(0, min(width, height) // 2, 18):
            fade = max(0, 45 - inset // 3)
            overlay_draw.rectangle((inset, inset, width - inset, height - inset), outline=(0, 0, 0, fade), width=16)

        return Image.alpha_composite(image, overlay)

    def _draw_glow(
        self,
        draw: Any,
        width: int,
        height: int,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        alpha: int,
    ) -> None:
        x_center, y_center = center
        draw.ellipse(
            (x_center - radius, y_center - radius, x_center + radius, y_center + radius),
            fill=(color[0], color[1], color[2], alpha),
        )

    def _build_palette(
        self, seed: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        palettes = [
            ((10, 14, 26), (31, 52, 86), (105, 175, 255), (234, 244, 255)),
            ((17, 11, 24), (64, 31, 78), (211, 115, 255), (250, 235, 255)),
            ((12, 20, 18), (31, 80, 84), (102, 207, 198), (236, 255, 248)),
            ((25, 18, 10), (86, 52, 24), (244, 175, 89), (255, 242, 212)),
            ((15, 17, 41), (48, 39, 96), (145, 131, 255), (243, 241, 255)),
        ]
        return palettes[seed % len(palettes)]

    def _blend(self, start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
        ratio = min(max(ratio, 0.0), 1.0)
        return (
            int(start[0] + (end[0] - start[0]) * ratio),
            int(start[1] + (end[1] - start[1]) * ratio),
            int(start[2] + (end[2] - start[2]) * ratio),
        )

    def _seed_from_text(self, text: str, scene_number: int) -> int:
        digest = hashlib.sha256(f"{scene_number}:{text}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _load_font(self, size: int) -> Any:
        from PIL import ImageFont

        windows_font_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates = [windows_font_root / "segoeui.ttf", windows_font_root / "arial.ttf"]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except Exception:
                    continue

        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

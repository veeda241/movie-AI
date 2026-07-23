from __future__ import annotations

from pathlib import Path

from video_lab.config import LabConfig
from video_lab.infer.research_generate import generate_research_video


def generate_finetune_video(
    prompt: str,
    *,
    seed: int = 0,
    steps: int = 20,
    frames: int | None = None,
    fps: int | None = None,
    height: int | None = None,
    width: int | None = None,
    log_fn=None,
) -> str:
    """
    Prefer CogVideoX+LoRA when available; otherwise fall back to research DiT generator
    so the Gradio Generate tab always returns an MP4.
    """
    cfg = LabConfig()
    lora_meta = cfg.lora_dir / "lora_meta.pt"
    frames = frames or cfg.frames
    fps = fps or cfg.fps
    height = height or cfg.height
    width = width or cfg.width
    try:
        import torch
        from diffusers import CogVideoXPipeline

        if not torch.cuda.is_available():
            if log_fn:
                log_fn("CUDA not available — using research DiT fallback for generation.")
            return generate_research_video(
                prompt,
                seed=seed,
                steps=steps,
                frames=frames,
                fps=fps,
                height=height,
                width=width,
                log_fn=log_fn,
            )

        pipe = CogVideoXPipeline.from_pretrained(cfg.base_t2v_model, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        if lora_meta.exists() and log_fn:
            log_fn(f"LoRA meta present at {lora_meta} (full adapter load depends on Diffusers export).")
        result = pipe(prompt=prompt, num_inference_steps=min(steps, 30), guidance_scale=6.0)
        frames_pil = result.frames[0]
        import numpy as np
        from PIL import Image

        from video_lab.utils.video_io import write_rgb_video

        arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames_pil], axis=0)
        # Optionally resample length/resolution toward requested settings
        if arr.shape[0] != frames or arr.shape[1] != height or arr.shape[2] != width:
            resized = []
            for i in range(frames):
                src_i = min(int(i * arr.shape[0] / max(frames, 1)), arr.shape[0] - 1)
                img = Image.fromarray(arr[src_i]).resize((width, height), Image.BILINEAR)
                resized.append(np.asarray(img, dtype=np.uint8))
            arr = np.stack(resized, axis=0)
        out = cfg.samples_dir / f"finetune_{seed}_{frames}f_{fps}fps.mp4"
        write_rgb_video(arr, out, fps=fps)
        if log_fn:
            log_fn(f"Wrote {out} ({frames / float(fps):.2f}s)")
        return str(out)
    except Exception as exc:
        if log_fn:
            log_fn(f"CogVideoX path unavailable ({exc}); using research DiT fallback.")
        return generate_research_video(
            prompt,
            seed=seed,
            steps=steps,
            frames=frames,
            fps=fps,
            height=height,
            width=width,
            log_fn=log_fn,
        )

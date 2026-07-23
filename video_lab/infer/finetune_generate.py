"""CogVideoX + LoRA generate (experimental). Falls back to research DiT if unavailable."""

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
    """Prefer CogVideoX+LoRA when CUDA + adapter exist; else research DiT."""
    cfg = LabConfig()
    frames = int(frames or cfg.frames)
    fps = int(fps or cfg.fps)
    height = int(height or cfg.height)
    width = int(width or cfg.width)
    adapter_dir = cfg.lora_dir / "lora_adapter"
    meta_path = cfg.lora_dir / "lora_meta.pt"

    def _fallback(reason: str) -> str:
        if log_fn:
            log_fn(f"{reason} — using own-model DiT fallback.")
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

    try:
        import torch

        if not torch.cuda.is_available():
            return _fallback("CUDA not available for CogVideoX")

        from diffusers import CogVideoXPipeline

        base = cfg.base_t2v_model
        if log_fn:
            log_fn(f"Loading CogVideoX: {base}")
        pipe = CogVideoXPipeline.from_pretrained(base, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            try:
                pipe.transformer.load_adapter(str(adapter_dir))
                if hasattr(pipe.transformer, "set_adapter"):
                    pipe.transformer.set_adapter("default")
                if log_fn:
                    log_fn(f"Loaded LoRA adapter: {adapter_dir}")
            except Exception as exc:
                # PEFT path used at train time
                try:
                    from peft import PeftModel

                    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(adapter_dir))
                    if log_fn:
                        log_fn(f"Loaded PEFT LoRA adapter: {adapter_dir}")
                except Exception as exc2:
                    if log_fn:
                        log_fn(f"Could not load LoRA adapter ({exc} / {exc2}); base CogVideoX only.")
        elif meta_path.exists() and log_fn:
            log_fn(f"Found {meta_path} but no adapter folder at {adapter_dir} — base model only.")

        result = pipe(
            prompt=prompt,
            num_inference_steps=min(int(steps), 30),
            guidance_scale=6.0,
        )
        frames_pil = result.frames[0]
        import numpy as np
        from PIL import Image

        from video_lab.utils.video_io import write_rgb_video

        arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames_pil], axis=0)
        if arr.shape[0] != frames or arr.shape[1] != height or arr.shape[2] != width:
            resized = []
            for i in range(frames):
                src_i = min(int(i * arr.shape[0] / max(frames, 1)), arr.shape[0] - 1)
                img = Image.fromarray(arr[src_i]).resize((width, height), Image.BILINEAR)
                resized.append(np.asarray(img, dtype=np.uint8))
            arr = np.stack(resized, axis=0)
        out = cfg.samples_dir / f"finetune_{seed}_{frames}f_{fps}fps.mp4"
        cfg.samples_dir.mkdir(parents=True, exist_ok=True)
        write_rgb_video(arr, out, fps=fps)
        if log_fn:
            log_fn(f"Wrote {out} ({frames / float(fps):.2f}s)")
        return str(out)
    except Exception as exc:
        return _fallback(f"CogVideoX path unavailable ({exc})")

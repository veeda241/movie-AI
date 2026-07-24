"""CogVideoX + LoRA generate (experimental)."""

from __future__ import annotations

from pathlib import Path

from video_lab.config import LabConfig


def _cogvideox_frames(n: int) -> int:
    """CogVideoX requires frame counts of the form 8N+1 (9, 17, 25, 49, …)."""
    n = max(9, int(n))
    return ((n - 1) // 8) * 8 + 1


def generate_finetune_video(
    prompt: str,
    *,
    seed: int = 0,
    steps: int = 50,
    frames: int | None = None,
    fps: int | None = None,
    height: int | None = None,
    width: int | None = None,
    use_lora: bool = True,
    allow_fallback: bool = False,
    log_fn=None,
) -> str:
    """Generate with CogVideoX (+ optional LoRA). Does not silently use research DiT."""
    cfg = LabConfig()
    adapter_dir = cfg.lora_dir / "lora_adapter"
    meta_path = cfg.lora_dir / "lora_meta.pt"

    # CogVideoX native defaults look decent; training meta can override if present.
    base = cfg.base_t2v_model
    height = 480 if height is None else int(height)
    width = 720 if width is None else int(width)
    frames = 49 if frames is None else _cogvideox_frames(frames)
    fps = int(fps or 8)
    height = max(64, height - (height % 16))
    width = max(64, width - (width % 16))

    if meta_path.exists():
        try:
            import torch as _torch

            saved = _torch.load(meta_path, map_location="cpu", weights_only=False)
            meta = saved.get("meta") or {}
            if meta.get("base_model"):
                base = meta["base_model"]
        except Exception as exc:
            if log_fn:
                log_fn(f"Could not read {meta_path}: {exc}")

    def _fallback(reason: str) -> str:
        if not allow_fallback:
            raise RuntimeError(
                f"{reason}\n"
                "Refusing research-DiT fallback (that path produces gray mush). "
                "Fix CogVideoX/LoRA load, or pass allow_fallback=True."
            )
        if log_fn:
            log_fn(f"{reason} — using own-model DiT fallback.")
        from video_lab.infer.research_generate import generate_research_video

        return generate_research_video(
            prompt,
            seed=seed,
            steps=steps,
            frames=max(8, frames // 4 * 4),
            fps=fps,
            height=min(height, 256),
            width=min(width, 256),
            log_fn=log_fn,
        )

    try:
        import torch
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
    except Exception as exc:
        return _fallback(f"CogVideoX imports failed ({exc})")

    if not torch.cuda.is_available():
        return _fallback("CUDA not available for CogVideoX")

    if log_fn:
        log_fn(f"Loading CogVideoX: {base}")
        log_fn(f"Infer size={width}x{height} frames={frames} steps={min(int(steps), 50)} seed={seed}")

    pipe = CogVideoXPipeline.from_pretrained(base, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    loaded_lora = False
    if use_lora and adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        # Training saves via PEFT — load that way first.
        try:
            from peft import PeftModel

            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(adapter_dir))
            loaded_lora = True
            if log_fn:
                log_fn(f"Loaded PEFT LoRA adapter: {adapter_dir}")
        except Exception as exc:
            try:
                pipe.transformer.load_adapter(str(adapter_dir))
                if hasattr(pipe.transformer, "set_adapter"):
                    pipe.transformer.set_adapter("default")
                loaded_lora = True
                if log_fn:
                    log_fn(f"Loaded LoRA adapter: {adapter_dir}")
            except Exception as exc2:
                if log_fn:
                    log_fn(f"Could not load LoRA ({exc} / {exc2}); generating with base only.")
    elif use_lora and log_fn:
        log_fn(f"No adapter at {adapter_dir} — generating with base CogVideoX only.")

    if log_fn and use_lora and not loaded_lora:
        log_fn("WARNING: LoRA not applied; output is base CogVideoX.")

    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=min(max(int(steps), 20), 50),
        num_frames=frames,
        height=height,
        width=width,
        guidance_scale=6.0,
        generator=generator,
    )
    frames_pil = result.frames[0]

    cfg.samples_dir.mkdir(parents=True, exist_ok=True)
    tag = "lora" if loaded_lora else "base"
    out = cfg.samples_dir / f"finetune_{tag}_{seed}_{frames}f_{width}x{height}.mp4"
    export_to_video(frames_pil, str(out), fps=fps)
    if log_fn:
        log_fn(f"Wrote {out} ({frames / float(fps):.2f}s @ {width}x{height}, lora={loaded_lora})")
    return str(out)

"""Wan2.1-T2V-1.3B + LoRA generate (experimental)."""

from __future__ import annotations

from video_lab.config import LabConfig


def _wan_frames(n: int) -> int:
    """Wan temporal constraint: latent frames = (T-1)//4 + 1 (VAE temporal
    downsample is 4), so T must be 4N+1 (5, 9, 13, ... 81)."""
    n = max(5, int(n))
    return ((n - 1) // 4) * 4 + 1


def _patch_wan_vae_config(vae):
    """The diffusers-format Wan2.1-T2V-1.3B-Diffusers checkpoint was saved with
    diffusers 0.33, which predates AutoencoderKLWan.scale_factor_temporal /
    scale_factor_spatial config keys. diffusers >=0.39's WanPipeline reads those
    at __init__ and crashes with "FrozenDict has no attribute scale_factor_temporal".
    Inject the Wan defaults (4 temporal / 8 spatial) if missing so the pipeline
    can construct safely."""
    for key, default in (("scale_factor_temporal", 4), ("scale_factor_spatial", 8)):
        if key not in vae.config:
            vae.config[key] = default
            vae.register_to_config(**{key: default})
    return vae


def _build_pipeline(base: str, dtype):
    """Load Wan components from the diffusers-format checkpoint and assemble the
    pipeline. We avoid WanPipeline.from_pretrained so we can patch the 0.33-era
    VAE config before the pipeline reads scale_factor_* at __init__."""
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from transformers import T5TokenizerFast, UMT5EncoderModel

    vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    _patch_wan_vae_config(vae)
    transformer = WanTransformer3DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=dtype
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = T5TokenizerFast.from_pretrained(base, subfolder="tokenizer")

    # Wan uses a flow-matching scheduler. UniPCMultistepScheduler (the value in
    # the repo's model_index.json) also works; FlowMatchEulerDiscreteScheduler is
    # the type WanPipeline.__init__ expects in diffusers 0.39 and is the default
    # for Wan2.1 T2V.
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=8.0,
    )

    return WanPipeline(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )


def generate_finetune_video(
    prompt: str,
    *,
    seed: int = 0,
    steps: int = 30,
    frames: int | None = None,
    fps: int | None = None,
    height: int | None = None,
    width: int | None = None,
    use_lora: bool = True,
    log_fn=None,
) -> str:
    """Generate with Wan2.1-T2V-1.3B (+ optional LoRA)."""
    import torch
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan generation.")

    cfg = LabConfig()
    adapter_dir = cfg.lora_dir / "lora_adapter"
    meta_path = cfg.lora_dir / "lora_meta.pt"

    def _is_wan_base(s: str) -> bool:
        s = (s or "").lower()
        return "wan" in s

    base = cfg.base_t2v_model
    height = cfg.wan_height if height is None else int(height)
    width = cfg.wan_width if width is None else int(width)
    frames = cfg.wan_frames if frames is None else _wan_frames(frames)
    fps = int(fps or cfg.wan_fps)
    # Enforce 16-divisibility (VAE 8x downsample, transformer patch 2x2)
    height = max(64, height - (height % 16))
    width = max(64, width - (width % 16))

    if meta_path.exists():
        try:
            saved = torch.load(meta_path, map_location="cpu", weights_only=False)
            meta = saved.get("meta") or {}
            meta_base = str(meta.get("base_model") or "")
            if meta_base and _is_wan_base(meta_base):
                base = meta_base
            elif meta_base and log_fn:
                # Ignore stale meta from another model family (e.g. old CogVideoX run)
                log_fn(f"Ignoring stale lora_meta base_model={meta_base!r} (not a Wan model); using {base}")
        except Exception as exc:
            if log_fn:
                log_fn(f"Could not read {meta_path}: {exc}")

    if log_fn:
        log_fn(f"Loading Wan: {base}")
        log_fn(f"Infer size={width}x{height} frames={frames} steps={min(int(steps), 50)} seed={seed}")

    if not _is_wan_base(base):
        raise RuntimeError(
            f"base_t2v_model={base!r} is not a Wan checkpoint. WanPipeline only "
            f"supports Wan-AI/Wan2.1-* (e.g. {cfg.base_t2v_model}). CogVideoX and "
            f"other model families cannot be loaded here; retrain your LoRA against "
            f"a Wan base model or pick a Wan checkpoint."
        )

    pipe = _build_pipeline(base, torch.bfloat16)
    pipe.enable_model_cpu_offload()

    loaded_lora = False
    if use_lora and adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
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
        log_fn(f"No adapter at {adapter_dir} — generating with base Wan only.")

    if log_fn and use_lora and not loaded_lora:
        log_fn("WARNING: LoRA not applied; output is base Wan.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Wan recommends guidance_scale=6 for the 1.3B model
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

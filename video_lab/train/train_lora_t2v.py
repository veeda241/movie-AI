"""Wan2.1-T2V-1.3B LoRA fine-tune (Diffusers + PEFT), 16GB-friendly offload."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch

from video_lab.config import LabConfig
from video_lab.data.dataset import VideoManifestDataset


def _log(msg: str, log_fn=None) -> None:
    if log_fn:
        log_fn(msg)
    else:
        print(msg, flush=True)


def _cuda_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _wan_frames(n: int) -> int:
    """Wan temporal constraint: latent frames = (T-1)//4 + 1 (VAE temporal
    downsample is 4). So T must be of the form 4N+1 (5, 9, 13, ... 81, ...).
    Wan2.1 native uses 81 frames (5s @ 16fps)."""
    n = max(5, int(n))
    return ((n - 1) // 4) * 4 + 1


def _wan_image_size(h: int, w: int) -> int:
    """Wan VAE spatial downsample is 8 and transformer patch is 2x2, so H and W
    must be divisible by 16."""
    h = max(64, int(h) - (int(h) % 16))
    w = max(64, int(w) - (int(w) % 16))
    return h, w


def _load_components(base_model: str, dtype):
    """Load the four Wan pipeline components from the diffusers-format checkpoint."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    from transformers import T5TokenizerFast, UMT5EncoderModel

    vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
    transformer = WanTransformer3DModel.from_pretrained(
        base_model, subfolder="transformer", torch_dtype=dtype
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer")
    # UniPCMultistepScheduler is the scheduler registered in the Wan model_index.json
    scheduler = UniPCMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
    return vae, transformer, text_encoder, tokenizer, scheduler


def _encode_text(text_encoder, tokenizer, captions, max_len: int, device, dtype):
    """Mirror WanPipeline.encode_prompt: pool, pad to max_len with zeros."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    if isinstance(captions, str):
        captions = [captions]
    captions = [prompt_clean(c) for c in captions]
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
    # trim each row to its real sequence length, then zero-pad back to max_len
    prompt_embeds = [u[: v.item()] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )
    return prompt_embeds


def _normalize_latents(vae, latents):
    """Wan maps the VAE latent space to the model prior via (latents - mean) * std,
    where mean/std come from vae.config.latents_mean / latents_std. (Inverse of
    the WanPipeline decode-time transform.)"""
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    mean = mean.to(latents.device, dtype=latents.dtype)
    std = std.to(latents.device, dtype=latents.dtype)
    return (latents - mean) * std


def train_lora_t2v(
    *,
    manifest_path: Path | None = None,
    base_model: str | None = None,
    steps: int | None = None,
    rank: int | None = None,
    lr: float | None = None,
    height: int | None = None,
    width: int | None = None,
    frames: int | None = None,
    low_vram: bool | None = None,
    log_fn=None,
) -> Path:
    """
    Wan2.1-T2V-1.3B LoRA fine-tuning using Diffusers + PEFT.

    On ~16GB GPUs the umt5 text encoder stays on CPU and the transformer + LoRA
    stays resident on CUDA (VAE encode moves to CPU in low_vram). The 1.3B model
    needs only ~8GB VRAM for inference; LoRA training is feasible on 16GB.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = LabConfig()
    manifest_path = Path(manifest_path or cfg.manifest_path)
    base_model = base_model or cfg.base_t2v_model
    steps = steps or cfg.lora_steps
    rank = rank or cfg.lora_rank
    lr = lr or 1e-4

    height = int(height) if height is not None else cfg.wan_height
    width = int(width) if width is not None else cfg.wan_width
    frames = int(frames) if frames is not None else cfg.wan_frames
    height, width = _wan_image_size(height, width)
    frames = _wan_frames(frames)
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)

    vram_gb = _cuda_gb()
    if low_vram is None:
        low_vram = vram_gb > 0 and vram_gb < 20.0

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if use_amp else torch.float32

    _log(
        f"Wan LoRA training: model={base_model} steps={steps} rank={rank} lr={lr} device={device}",
        log_fn,
    )
    _log(f"Size={width}x{height} frames={frames} low_vram={low_vram} vram~{vram_gb:.1f}GB", log_fn)
    _log(f"Manifest: {manifest_path}", log_fn)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Wan fine-tuning. The 1.3B model + umt5 encoder "
            "cannot train on CPU."
        )

    _log("Loading Wan components...", log_fn)
    vae, transformer, text_encoder, tokenizer, _scheduler = _load_components(base_model, dtype)
    # Ensure VAE config has scale_factor_* (older 0.33-era checkpoint omits them).
    for _key, _default in (("scale_factor_temporal", 4), ("scale_factor_spatial", 8)):
        if _key not in vae.config:
            vae.config[_key] = _default
            vae.register_to_config(**{_key: _default})

    # Memory plan: only transformer on GPU when low_vram
    if low_vram:
        _log("low_vram: umt5 + VAE on CPU; transformer+LoRA on CUDA", log_fn)
        vae.to("cpu", dtype=torch.float32)
        text_encoder.to("cpu", dtype=torch.float32)
        transformer.to(device, dtype=dtype)
    else:
        vae.to(device, dtype=dtype)
        text_encoder.to(device, dtype=dtype)
        transformer.to(device, dtype=dtype)
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    _log(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}", log_fn)

    from peft import LoraConfig, get_peft_model

    # LoRA target every transformer block by suffix match:
    #   - self/cross attention:  to_q, to_k, to_v, to_out.0
    #   - FFN (FeedForward):     ffn.net.0.proj (in-projection), ffn.net.2 (out-projection)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights=True,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    from torch.utils.data import DataLoader

    dataset = VideoManifestDataset(
        manifest_path,
        frames=frames,
        height=height,
        width=width,
        bucket="square_256" if height == 256 and width == 256 else None,
        letterbox=True,
    )
    batch_size = 1
    _log(f"Dataset: {len(dataset)} clips, batch_size={batch_size}", log_fn)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset from manifest: {manifest_path}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        (p for p in transformer.parameters() if p.requires_grad),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")

    transformer.train()
    global_step = 0
    losses: list[float] = []
    start_time = time.time()
    max_text_len = 226  # Wan default max_sequence_length

    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break

            optimizer.zero_grad(set_to_none=True)

            captions = batch["caption"]
            videos = batch["video"]  # (1, 3, T, H, W) in [-1, 1]

            # --- text embeds ---
            text_device = "cpu" if low_vram else device
            text_encoder.to(text_device)  # ensure encoder on right device for encode
            prompt_embeds = _encode_text(
                text_encoder, tokenizer, list(captions), max_text_len, text_device, dtype
            )
            prompt_embeds = prompt_embeds.to(device, dtype=dtype).detach()

            # --- VAE encode -> Wan latent prior ---
            with torch.no_grad():
                if low_vram:
                    vae.to("cpu", dtype=torch.float32)
                    vid = videos.float().cpu()
                    latents = vae.encode(vid).latents
                    latents = _normalize_latents(vae, latents).to(device, dtype=dtype)
                    vae.to("cpu")  # keep VAE on CPU between steps in low_vram
                else:
                    vid = videos.to(device, dtype=dtype)
                    latents = vae.encode(vid).latents
                    latents = _normalize_latents(vae, latents)

            del videos
            if device == "cuda":
                torch.cuda.empty_cache()

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Wan uses flow-matching: train a velocity field. Sample t in [0,1].
            timesteps = torch.rand((bsz,), device=device)
            t_exp = timesteps.view(bsz, *[1] * (latents.ndim - 1))
            # Flow-matching: x_t = (1 - t) * x0 + t * noise ; target v = noise - x0
            noisy_latents = (1.0 - t_exp) * latents + t_exp * noise

            def _forward():
                out = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                target = noise - latents
                loss = ((out.float() - target.float()) ** 2).mean()
                return loss

            try:
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = _forward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in transformer.parameters() if p.requires_grad), 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = _forward()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in transformer.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during Wan LoRA step. Try:\n"
                    "  1) --frames 17 --height 256 --width 256 --rank 8\n"
                    "  2) Close other GPU apps\n"
                    "  3) A 24GB+ GPU for full 832x480x81 training"
                ) from None

            losses.append(float(loss.detach().item()))
            global_step += 1
            cosine.step()

            if global_step % 10 == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)
                lr_now = optimizer.param_groups[0]["lr"]
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                _log(
                    f"  Step {global_step}/{steps} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                    f"elapsed={elapsed:.0f}s | mem={mem:.2f}GB",
                    log_fn,
                )

            if global_step % 50 == 0:
                save_path = cfg.lora_dir / f"lora_step_{global_step}"
                transformer.save_pretrained(save_path)
                _log(f"Saved intermediate adapter: {save_path}", log_fn)

            if device == "cuda":
                torch.cuda.empty_cache()

        if global_step >= steps:
            break

    final_path = cfg.lora_dir / "lora_adapter"
    transformer.save_pretrained(final_path)
    lora_config.save_pretrained(final_path)

    meta = {
        "base_model": base_model,
        "rank": rank,
        "steps": steps,
        "lr": lr,
        "height": height,
        "width": width,
        "frames": frames,
        "low_vram": low_vram,
        "batch_size": batch_size,
        "final_loss": float(sum(losses[-100:]) / max(len(losses[-100:]), 1)),
        "total_steps_completed": global_step,
        "manifest": str(manifest_path),
        "lora_adapter_path": str(final_path),
    }
    meta_path = cfg.lora_dir / "lora_meta.pt"
    torch.save({"meta": meta, "lora_config": lora_config.to_dict()}, meta_path)
    _log(f"Saved LoRA adapter: {final_path}", log_fn)
    _log(f"Saved LoRA meta: {meta_path}", log_fn)
    _log(f"Training complete in {time.time() - start_time:.0f}s", log_fn)
    return final_path

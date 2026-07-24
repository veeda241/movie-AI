"""CogVideoX LoRA fine-tune (Diffusers + PEFT) with 16GB-friendly offload."""

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


def _prepare_rotary(
    *,
    height: int,
    width: int,
    num_frames: int,
    transformer,
    device: torch.device,
):
    """Best-effort 3D RoPE for CogVideoX; returns None if unavailable."""
    try:
        from diffusers.models.embeddings import get_3d_rotary_pos_embed
    except Exception:
        return None

    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    model_config = base.config
    if not getattr(model_config, "use_rotary_positional_embeddings", False):
        return None

    vae_scale = 8
    patch_size = int(getattr(model_config, "patch_size", 2))
    attention_head_dim = int(getattr(model_config, "attention_head_dim", 64))
    grid_h = height // (vae_scale * patch_size)
    grid_w = width // (vae_scale * patch_size)
    if grid_h < 1 or grid_w < 1:
        return None

    try:
        return get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_h, grid_w),
            temporal_size=num_frames,
            device=device,
        )
    except TypeError:
        try:
            return get_3d_rotary_pos_embed(
                embed_dim=attention_head_dim,
                crops_coords=((0, 0), (grid_h, grid_w)),
                grid_size=(grid_h, grid_w),
                temporal_size=num_frames,
                device=device,
            )
        except Exception:
            return None
    except Exception:
        return None


def _cuda_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _cogvideox_frames(n: int) -> int:
    """Official CogVideoX constraint: frame count must be 8N+1 (9, 17, 25, 49, …)."""
    n = max(9, int(n))
    return ((n - 1) // 8) * 8 + 1


def train_lora_t2v(
    *,
    manifest_path: Path | None = None,
    base_model: str | None = None,
    steps: int | None = None,
    rank: int | None = None,
    lr: float | None = None,
    height: int = 256,
    width: int = 256,
    frames: int = 9,
    low_vram: bool | None = None,
    log_fn=None,
) -> Path:
    """
    CogVideoX LoRA fine-tuning using Diffusers + PEFT.

    On ~16GB GPUs, VAE + T5 stay on CPU and only the transformer+LoRA
    stays on CUDA (otherwise VAE encode OOMs with 5B weights resident).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = LabConfig()
    manifest_path = Path(manifest_path or cfg.manifest_path)
    base_model = base_model or cfg.base_t2v_model
    steps = steps or cfg.lora_steps
    rank = rank or cfg.lora_rank
    lr = lr or 1e-4
    height = max(64, int(height) - (int(height) % 16))
    width = max(64, int(width) - (int(width) % 16))
    frames = _cogvideox_frames(frames)
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)

    vram_gb = _cuda_gb()
    if low_vram is None:
        low_vram = vram_gb > 0 and vram_gb < 20.0

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_amp else torch.float32

    _log(f"LoRA training: model={base_model} steps={steps} rank={rank} lr={lr} device={device}", log_fn)
    _log(f"Size={width}x{height} frames={frames} low_vram={low_vram} vram~{vram_gb:.1f}GB", log_fn)
    _log(f"Manifest: {manifest_path}", log_fn)

    from diffusers import CogVideoXPipeline
    from peft import LoraConfig, get_peft_model

    _log("Loading base model...", log_fn)
    pipe = CogVideoXPipeline.from_pretrained(base_model, torch_dtype=dtype)

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    transformer = pipe.transformer
    # Drop pipeline wrapper so we can free unused refs
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Memory plan: only transformer on GPU when low_vram
    if low_vram and device == "cuda":
        _log("low_vram: VAE + text encoder on CPU; transformer+LoRA on CUDA", log_fn)
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

    _log(f"Model loaded. Transformer params: {sum(p.numel() for p in transformer.parameters()):,}", log_fn)

    try:
        import tiktoken  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CogVideoX tokenizer needs tiktoken. Install: pip install tiktoken sentencepiece"
        ) from exc

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
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
    batch_size = 1  # keep 1 for CogVideoX LoRA on consumer GPUs
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
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")

    transformer.train()
    global_step = 0
    losses: list[float] = []
    start_time = time.time()
    base_cfg = transformer.get_base_model().config
    max_text_len = int(getattr(base_cfg, "max_text_seq_length", 226))
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))

    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break

            optimizer.zero_grad(set_to_none=True)

            captions = batch["caption"]
            if isinstance(captions, str):
                captions = [captions]
            videos = batch["video"]  # (B, C, T, H, W) float CPU/GPU tensor

            # --- text embeds (CPU in low_vram) ---
            text_inputs = tokenizer(
                list(captions),
                padding="max_length",
                max_length=max_text_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_device = "cpu" if low_vram else device
            text_inputs = {k: v.to(text_device) for k, v in text_inputs.items()}
            with torch.no_grad():
                encoder_hidden_states = text_encoder(**text_inputs)[0]
                encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype).detach()

            # --- VAE encode (CPU in low_vram to free GPU for transformer) ---
            with torch.no_grad():
                if low_vram:
                    vid = videos.float().cpu()
                    latent_dist = vae.encode(vid).latent_dist
                    latents = latent_dist.sample() * scaling
                    latents = latents.to(device, dtype=dtype)
                else:
                    vid = videos.to(device, dtype=dtype)
                    latent_dist = vae.encode(vid).latent_dist
                    latents = latent_dist.sample() * scaling
                # Transformer expects [B, F, C, H, W]
                latents = latents.permute(0, 2, 1, 3, 4).contiguous()

            del videos
            if device == "cuda":
                torch.cuda.empty_cache()

            noise = torch.randn_like(latents)
            bsz, num_frames, _c, _lh, _lw = latents.shape
            timesteps = torch.randint(
                0, int(scheduler.config.num_train_timesteps), (bsz,), device=device
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            image_rotary_emb = _prepare_rotary(
                height=height,
                width=width,
                num_frames=num_frames,
                transformer=transformer,
                device=torch.device(device),
            )

            def _forward():
                out = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(out, noisy_latents, timesteps)
                alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=model_pred.dtype)
                weights = 1.0 / (1.0 - alphas_cumprod[timesteps]).clamp_min(1e-4)
                while weights.ndim < model_pred.ndim:
                    weights = weights.unsqueeze(-1)
                return torch.mean(
                    (weights * (model_pred - latents) ** 2).reshape(bsz, -1), dim=1
                ).mean()

            try:
                if use_amp and device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
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
                    "CUDA OOM during LoRA step. Try:\n"
                    "  1) --base-model THUDM/CogVideoX-2b\n"
                    "  2) --height 192 --width 192 --frames 9 --rank 8\n"
                    "  3) Close other GPU apps\n"
                    "  4) Use a 24GB+ GPU for CogVideoX-5b"
                ) from None

            losses.append(float(loss.detach().item()))
            global_step += 1
            scheduler_t.step()

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

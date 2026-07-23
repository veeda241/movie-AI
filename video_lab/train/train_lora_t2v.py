"""CogVideoX LoRA fine-tune (Diffusers + PEFT), aligned with official training call signature."""

from __future__ import annotations

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

    cfg = transformer.config if hasattr(transformer, "config") else getattr(transformer, "base_model", transformer)
    # unwrap peft
    base = transformer
    if hasattr(transformer, "get_base_model"):
        base = transformer.get_base_model()
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
        # Newer diffusers API
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_h, grid_w),
            temporal_size=num_frames,
            device=device,
        )
        return freqs_cos, freqs_sin
    except TypeError:
        try:
            freqs = get_3d_rotary_pos_embed(
                embed_dim=attention_head_dim,
                crops_coords=((0, 0), (grid_h, grid_w)),
                grid_size=(grid_h, grid_w),
                temporal_size=num_frames,
                device=device,
            )
            return freqs
        except Exception:
            return None
    except Exception:
        return None


def train_lora_t2v(
    *,
    manifest_path: Path | None = None,
    base_model: str | None = None,
    steps: int | None = None,
    rank: int | None = None,
    lr: float | None = None,
    log_fn=None,
) -> Path:
    """
    CogVideoX LoRA fine-tuning using Diffusers + PEFT.

    Call signature matches Hugging Face `train_cogvideox_lora.py`:
    keyword-only `hidden_states` / `encoder_hidden_states` / `timestep`,
    latents in `[B, F, C, H, W]`, velocity loss.
    """
    cfg = LabConfig()
    manifest_path = Path(manifest_path or cfg.manifest_path)
    base_model = base_model or cfg.base_t2v_model
    steps = steps or cfg.lora_steps
    rank = rank or cfg.lora_rank
    lr = lr or 1e-4
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_amp else torch.float32

    _log(f"LoRA training: model={base_model} steps={steps} rank={rank} lr={lr} device={device}", log_fn)
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

    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    text_encoder.to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    transformer.to(device, dtype=dtype)
    _log(f"Model loaded. Transformer params: {sum(p.numel() for p in transformer.parameters()):,}", log_fn)

    try:
        import tiktoken  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CogVideoX tokenizer needs tiktoken. Install: pip install tiktoken sentencepiece"
        ) from exc
    _log("Dependencies OK (tiktoken, sentencepiece)", log_fn)

    # Official CogVideoX LoRA targets (attention projections)
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

    train_h, train_w, train_f = 256, 256, 8
    dataset = VideoManifestDataset(
        manifest_path,
        frames=train_f,
        height=train_h,
        width=train_w,
        bucket="square_256",
        letterbox=True,
    )
    total_vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
    batch_size = 2 if total_vram > 20e9 else 1
    _log(f"Total VRAM: {total_vram / 1e9:.1f}GB, batch_size={batch_size}", log_fn)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    transformer.train()
    global_step = 0
    losses: list[float] = []
    start_time = time.time()
    max_text_len = int(getattr(transformer.get_base_model().config, "max_text_seq_length", 226))

    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break

            optimizer.zero_grad(set_to_none=True)

            captions = batch["caption"]
            if isinstance(captions, str):
                captions = [captions]
            videos = batch["video"].to(device, dtype=dtype)  # (B, C, T, H, W)

            text_inputs = tokenizer(
                list(captions),
                padding="max_length",
                max_length=max_text_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            with torch.no_grad():
                encoder_hidden_states = text_encoder(**text_inputs)[0].detach()

            with torch.no_grad():
                # CogVideoX VAE expects [B, C, F, H, W]
                latent_dist = vae.encode(videos).latent_dist
                latents = latent_dist.sample() * vae.config.scaling_factor
                # Transformer expects [B, F, C, H, W]
                latents = latents.permute(0, 2, 1, 3, 4).contiguous()

            noise = torch.randn_like(latents)
            bsz, num_frames, _c, _lh, _lw = latents.shape
            timesteps = torch.randint(
                0, int(scheduler.config.num_train_timesteps), (bsz,), device=device
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            image_rotary_emb = _prepare_rotary(
                height=train_h,
                width=train_w,
                num_frames=num_frames,
                transformer=transformer,
                device=torch.device(device),
            )

            def _forward():
                # IMPORTANT: keyword args only — positional conflicts with PEFT wrapper
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
                target = latents
                return torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1).mean()

            if use_amp and scaler is not None:
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

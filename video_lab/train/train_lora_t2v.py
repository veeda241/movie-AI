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
    CogVideoX-5B LoRA fine-tuning using Diffusers + PEFT.

    Loads the base model, applies LoRA to the transformer block,
    trains on video-caption pairs from manifest, saves adapter weights.
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

    # ------------------------------------------------------------------ #
    # 1. Load base pipeline (transformer only to save VRAM)
    # ------------------------------------------------------------------ #
    from diffusers import CogVideoXPipeline
    from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
    from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

    _log("Loading base model...", log_fn)
    pipe = CogVideoXPipeline.from_pretrained(base_model, torch_dtype=dtype)

    # Move VAE + text encoder to CPU or offload to save VRAM during training
    vae: AutoencoderKLCogVideoX = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    transformer: CogVideoXTransformer3DModel = pipe.transformer

    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    text_encoder.to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    transformer.to(device, dtype=dtype)

    _log(f"Model loaded. Transformer params: {sum(p.numel() for p in transformer.parameters()):,}", log_fn)

    # ------------------------------------------------------------------ #
    # 2. Check dependencies
    # ------------------------------------------------------------------ #
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        raise ImportError(
            "CogVideoX-5B tokenizer requires tiktoken. "
            "Install it: pip install tiktoken sentencepiece"
        )
    _log("Dependencies OK (tiktoken, sentencepiece)", log_fn)

    # ------------------------------------------------------------------ #
    # 2. Apply LoRA to transformer
    # ------------------------------------------------------------------ #
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "ff.net.0.proj", "ff.net.2"],
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Enable gradient checkpointing for VRAM
    transformer.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ #
    # 3. Dataset + DataLoader
    # ------------------------------------------------------------------ #
    from torch.utils.data import DataLoader

    # Use first available clip params from manifest
    dataset = VideoManifestDataset(
        manifest_path,
        frames=8,
        height=256,
        width=256,
        bucket="square_256",
        letterbox=True,
    )

    # Determine batch size based on VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
    batch_size = 2 if total_vram > 20e9 else 1  # >20GB can handle batch_size=2 with LoRA
    _log(f"Total VRAM: {total_vram/1e9:.1f}GB, batch_size={batch_size}", log_fn)
    _log(f"Dataset: {len(dataset)} clips, batch_size={batch_size}", log_fn)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ------------------------------------------------------------------ #
    # 4. Optimizer + Scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ------------------------------------------------------------------ #
    # 5. Training Loop
    # ------------------------------------------------------------------ #
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    transformer.train()
    global_step = 0
    losses: list[float] = []
    start_time = time.time()

    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break

            optimizer.zero_grad()

            captions = batch["caption"]
            videos = batch["video"].to(device, dtype=dtype)  # (B, C, T, H, W)

            # Encode captions to embeddings (once per batch, cache)
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=getattr(tokenizer, "model_max_length", 226),
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(**text_inputs)[0].detach()

            # Encode video to latents (VAE stays on device but uses no grad)
            with torch.no_grad():
                latents = vae.encode(videos).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()

            # Add noise to latents
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            del latents  # free original latents

            # Predict noise
            if use_amp:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    noise_pred = transformer(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            else:
                noise_pred = transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            # Backward: common path for both AMP and non-AMP
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())
            global_step += 1

            # Step the LR scheduler
            scheduler_t.step()

            if global_step % 10 == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)
                lr_now = optimizer.param_groups[0]["lr"]
                _log(
                    f"  Step {global_step}/{steps} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                    f"elapsed={elapsed:.0f}s | mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                    log_fn,
                )

            if global_step % 50 == 0:
                # Save intermediate
                save_path = cfg.lora_dir / f"lora_step_{global_step}"
                transformer.save_pretrained(save_path)
                _log(f"Saved intermediate adapter: {save_path}", log_fn)

        if global_step >= steps:
            break

    # ------------------------------------------------------------------ #
    # 6. Save final LoRA adapter
    # ------------------------------------------------------------------ #
    final_path = cfg.lora_dir / "lora_adapter"
    transformer.save_pretrained(final_path)
    lora_config.save_pretrained(final_path)

    # Also save as a combined checkpoint for easy loading
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

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from video_lab.config import LabConfig
from video_lab.data.dataset import VideoManifestDataset
from video_lab.data.smoke import ensure_smoke_manifest
from video_lab.models.causal_vae import CausalVAE3D
from video_lab.models.dit import SpatioTemporalDiT
from video_lab.utils.device import get_device


def _q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, timesteps: int = 1000) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, timesteps, device=x0.device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    a = alpha_bar[t].view(-1, 1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def train_dit(
    *,
    manifest_path: Path | None = None,
    vae_ckpt: Path | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    min_aesthetic: float | None = None,
    dit_size: str | None = None,
    bucket: str | None = None,
    train_stage: str | None = None,
    use_amp: bool = True,
    log_fn=None,
) -> Path:
    cfg = LabConfig()
    if dit_size:
        cfg.dit_size = dit_size
    manifest_path = Path(manifest_path or cfg.manifest_path)
    frames = max(8, (frames or cfg.frames) // 4 * 4)
    height = max(32, (height or cfg.height) // 8 * 8)
    width = max(32, (width or cfg.width) // 8 * 8)

    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        ensure_smoke_manifest(manifest_path, frames=frames, size=max(height, width))
        if log_fn:
            log_fn(f"Created smoke manifest at {manifest_path}")
    elif log_fn:
        log_fn(f"Using existing manifest {manifest_path} (labels preserved)")

    device = get_device()
    steps = steps or cfg.dit_steps
    batch_size = batch_size or cfg.batch_size
    lr = lr or getattr(cfg, "dit_lr", cfg.lr)
    hidden, layers, heads = cfg.dit_dims()
    patch = cfg.dit_patch_size()
    amp = bool(use_amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if log_fn:
        log_fn(f"DiT size={cfg.dit_size} hidden={hidden} layers={layers} patch={patch} amp={amp}")

    vae_base = int(getattr(cfg, "vae_base_channels", 48))
    vae = CausalVAE3D(latent_ch=cfg.vae_latent_channels, base=vae_base).to(device)
    vae_path = Path(vae_ckpt or (cfg.vae_dir / "vae_last.pt"))
    if vae_path.exists():
        state = torch.load(vae_path, map_location=device, weights_only=False)
        meta = state.get("config") or {}
        vae_base = int(meta.get("base", vae_base))
        vae = CausalVAE3D(latent_ch=cfg.vae_latent_channels, base=vae_base).to(device)
        missing, _ = vae.load_state_dict(state["model"], strict=False)
        if log_fn:
            log_fn(f"Loaded VAE: {vae_path}" + (f" (missing={len(missing)})" if missing else ""))
    else:
        if log_fn:
            log_fn("No VAE checkpoint; training DiT with random VAE encoder.")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit = SpatioTemporalDiT(
        latent_ch=cfg.vae_latent_channels,
        hidden=hidden,
        layers=layers,
        heads=heads,
        patch_size=patch,
    ).to(device)
    opt = torch.optim.AdamW(dit.parameters(), lr=lr)

    ds = VideoManifestDataset(
        manifest_path,
        frames=frames,
        height=height,
        width=width,
        min_aesthetic=float(min_aesthetic if min_aesthetic is not None else cfg.min_aesthetic),
        bucket=bucket,
        letterbox=True,
    )
    if log_fn:
        log_fn(f"DiT train set={len(ds)} stage={train_stage or cfg.train_stage}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    cfg.dit_dir.mkdir(parents=True, exist_ok=True)
    dit.train()
    step = 0
    pbar = tqdm(total=steps, desc="DiT")
    last_loss = 0.0
    while step < steps:
        for batch in loader:
            video = batch["video"].to(device)
            if video.ndim == 4:
                video = video.unsqueeze(0)
            captions = batch["caption"] if isinstance(batch["caption"], list) else [batch["caption"]]
            if torch.rand(1).item() < 0.1:
                captions = [""] * len(captions)
            if not torch.isfinite(video).all():
                continue
            with torch.no_grad():
                mean, _logvar = vae.encoder(video)
                z = mean
            if not torch.isfinite(z).all():
                continue
            noise = torch.randn_like(z)
            t = torch.randint(0, 1000, (z.shape[0],), device=device)
            noisy = _q_sample(z, t, noise)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = dit(noisy, t, captions)
                loss = torch.nn.functional.mse_loss(pred, noise)
            if not torch.isfinite(loss):
                if log_fn and step % 50 == 0:
                    log_fn(f"DiT skip non-finite loss at step {step}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.detach().float().item())
            step += 1
            pbar.update(1)
            if log_fn and step % max(1, steps // 10) == 0:
                log_fn(f"DiT step {step}/{steps} loss={last_loss:.4f}")
            if step >= steps:
                break
    pbar.close()

    ckpt = cfg.dit_dir / "dit_last.pt"
    torch.save(
        {
            "model": dit.state_dict(),
            "config": {
                "latent_ch": cfg.vae_latent_channels,
                "hidden": hidden,
                "layers": layers,
                "heads": heads,
                "dit_size": cfg.dit_size,
                "patch_size": list(patch),
                "frames": frames,
                "height": height,
                "width": width,
                "bucket": bucket,
                "stage": train_stage or cfg.train_stage,
                "steps": steps,
                "loss": last_loss,
                "min_aesthetic": float(min_aesthetic if min_aesthetic is not None else cfg.min_aesthetic),
            },
        },
        ckpt,
    )
    if log_fn:
        log_fn(f"Saved DiT {ckpt} (final loss={last_loss:.4f})")
    return ckpt

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from video_lab.config import LabConfig
from video_lab.data.dataset import VideoManifestDataset
from video_lab.data.smoke import ensure_smoke_manifest
from video_lab.models.causal_vae import CausalVAE3D, vae_loss
from video_lab.utils.device import get_device


def train_vae(
    *,
    manifest_path: Path | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    min_aesthetic: float | None = None,
    bucket: str | None = None,
    train_stage: str | None = None,
    use_amp: bool = True,
    log_fn=None,
) -> Path:
    cfg = LabConfig()
    manifest_path = Path(manifest_path or cfg.manifest_path)
    # Align dims to VAE strides (8 spatial, 4 temporal)
    frames = max(8, (frames or cfg.frames) // 4 * 4)
    height = max(32, (height or cfg.height) // 8 * 8)
    width = max(32, (width or cfg.width) // 8 * 8)

    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        ensure_smoke_manifest(manifest_path, frames=frames, size=max(height, width))
        if log_fn:
            log_fn(f"Created smoke manifest at {manifest_path} ({frames}f @ {height}x{width})")
    elif log_fn:
        log_fn(f"Using existing manifest {manifest_path} (labels preserved)")

    device = get_device()
    steps = steps or cfg.vae_steps
    batch_size = batch_size or cfg.batch_size
    lr = lr or getattr(cfg, "vae_lr", cfg.lr)
    base = int(getattr(cfg, "vae_base_channels", 48))
    amp = bool(use_amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

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
        log_fn(
            f"VAE train set={len(ds)} stage={train_stage or cfg.train_stage} "
            f"size={width}x{height}f{frames} amp={amp}"
        )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = CausalVAE3D(latent_ch=cfg.vae_latent_channels, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    cfg.vae_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="VAE")
    last_loss = 0.0
    while step < steps:
        for batch in loader:
            video = batch["video"].to(device)
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if not torch.isfinite(video).all():
                continue
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                recon, mean, logvar = model(video)
                loss = vae_loss(recon, video, mean, logvar)
            if not torch.isfinite(loss):
                if log_fn and step % 50 == 0:
                    log_fn(f"VAE skip non-finite loss at step {step}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.detach().float().item())
            step += 1
            pbar.update(1)
            if log_fn and step % max(1, steps // 10) == 0:
                log_fn(f"VAE step {step}/{steps} loss={last_loss:.4f}")
            if step >= steps:
                break
    pbar.close()

    ckpt = cfg.vae_dir / "vae_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "latent_ch": cfg.vae_latent_channels,
                "base": base,
                "spatial_compress": model.spatial_compress,
                "temporal_compress": model.temporal_compress,
                "frames": frames,
                "height": height,
                "width": width,
                "bucket": bucket,
                "stage": train_stage or cfg.train_stage,
                "steps": steps,
                "loss": last_loss,
            },
        },
        ckpt,
    )
    if log_fn:
        log_fn(f"Saved VAE {ckpt} (8xS/4xT, loss={last_loss:.4f})")
    return ckpt

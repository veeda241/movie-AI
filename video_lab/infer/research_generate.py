"""Research video generation from local VAE + DiT checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from video_lab.config import LabConfig
from video_lab.models.causal_vae import CausalVAE3D
from video_lab.models.dit import SpatioTemporalDiT
from video_lab.utils.device import get_device
from video_lab.utils.video_io import frames_from_tensor, write_rgb_video


def _load_models(cfg: LabConfig, device: torch.device, log_fn=None):
    vae_ckpt = cfg.vae_dir / "vae_last.pt"
    dit_ckpt = cfg.dit_dir / "dit_last.pt"
    train_frames, train_h, train_w = 16, 128, 128
    hidden, layers, heads = cfg.dit_dims()
    patch = cfg.dit_patch_size()
    vae_base = int(getattr(cfg, "vae_base_channels", 48))
    spatial_c, temporal_c = 8, 4

    if vae_ckpt.exists():
        vae_state = torch.load(vae_ckpt, map_location=device, weights_only=False)
        meta = vae_state.get("config") or {}
        vae_base = int(meta.get("base", vae_base))
        spatial_c = int(meta.get("spatial_compress", spatial_c))
        temporal_c = int(meta.get("temporal_compress", temporal_c))
        train_frames = int(meta.get("frames", train_frames))
        train_h = int(meta.get("height", train_h))
        train_w = int(meta.get("width", train_w))
        vae = CausalVAE3D(latent_ch=cfg.vae_latent_channels, base=vae_base).to(device)
        missing, _ = vae.load_state_dict(vae_state["model"], strict=False)
        if log_fn:
            log_fn(
                f"Loaded VAE ({temporal_c}xT/{spatial_c}xS, trained {train_frames}f @ {train_w}x{train_h})"
                + (f" missing={len(missing)}" if missing else "")
            )
    else:
        vae = CausalVAE3D(latent_ch=cfg.vae_latent_channels, base=vae_base).to(device)
        if log_fn:
            log_fn("No VAE checkpoint — random VAE.")

    if dit_ckpt.exists():
        dit_state = torch.load(dit_ckpt, map_location=device, weights_only=False)
        meta = dit_state.get("config") or {}
        hidden = int(meta.get("hidden", hidden))
        layers = int(meta.get("layers", layers))
        heads = int(meta.get("heads", heads))
        ps = meta.get("patch_size", list(patch))
        patch = tuple(int(x) for x in ps)
        train_frames = int(meta.get("frames", train_frames))
        train_h = int(meta.get("height", train_h))
        train_w = int(meta.get("width", train_w))
        dit = SpatioTemporalDiT(
            latent_ch=cfg.vae_latent_channels,
            hidden=hidden,
            layers=layers,
            heads=heads,
            patch_size=patch,
        ).to(device)
        missing, _ = dit.load_state_dict(dit_state["model"], strict=False)
        if log_fn:
            log_fn(f"Loaded DiT (patch={patch}, hidden={hidden})")
            if missing:
                log_fn("Some DiT weights missing — retrain DiT.")
    else:
        dit = SpatioTemporalDiT(
            latent_ch=cfg.vae_latent_channels,
            hidden=hidden,
            layers=layers,
            heads=heads,
            patch_size=patch,
        ).to(device)
        if log_fn:
            log_fn("No DiT checkpoint — random DiT.")

    train_frames = max(8, train_frames - (train_frames % temporal_c))
    train_h = max(32, train_h - (train_h % spatial_c))
    train_w = max(32, train_w - (train_w % spatial_c))
    return vae, dit, train_frames, train_h, train_w, spatial_c, temporal_c


def _latent_shape(vae: CausalVAE3D, frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    """(C, T', H', W') after VAE encode for B=1."""
    sc = int(getattr(vae, "spatial_compress", 8))
    tc = int(getattr(vae, "temporal_compress", 4))
    c = int(getattr(vae, "latent_ch", 4))
    return (c, max(1, frames // tc), max(1, height // sc), max(1, width // sc))


@torch.no_grad()
def _sample_chunk_u8(
    *,
    vae: CausalVAE3D,
    dit: SpatioTemporalDiT,
    prompt: str,
    chunk_frames: int,
    chunk_h: int,
    chunk_w: int,
    out_h: int,
    out_w: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    device: torch.device,
    chunk_idx: int = 0,
) -> np.ndarray:
    """Denoise from pure Gaussian noise (no smoke / reference prior)."""
    g = torch.Generator(device=device).manual_seed(seed + chunk_idx * 9973)
    c, zt, zh, zw = _latent_shape(vae, chunk_frames, chunk_h, chunk_w)

    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn((1, c, zt, zh, zw), device=device, generator=g)
    # Full trajectory from high noise (better text control than img2img-from-smoke)
    start_t = 999
    timesteps = torch.linspace(start_t, 0, steps, dtype=torch.long, device=device)

    for i, t_int in enumerate(timesteps):
        t = t_int.expand(1)
        eps_c = dit(x, t, [prompt])
        if cfg_scale and cfg_scale != 1.0:
            eps_u = dit(x, t, [""])
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        else:
            eps = eps_c
        ab = alpha_bar[t].view(-1, 1, 1, 1, 1)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)
        x0 = x0.clamp(-4, 4)
        if i + 1 < len(timesteps):
            ab_next = alpha_bar[timesteps[i + 1]].view(-1, 1, 1, 1, 1)
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        else:
            x = x0

    video = vae.decode(x)
    if video.shape[2] != chunk_frames or video.shape[-2:] != (out_h, out_w):
        video = torch.nn.functional.interpolate(
            video, size=(chunk_frames, out_h, out_w), mode="trilinear", align_corners=False
        )
    return frames_from_tensor(video)


def _crossfade_concat(chunks: list[np.ndarray], overlap: int = 2) -> np.ndarray:
    """Concatenate uint8 (T,H,W,3) chunks with a short linear crossfade."""
    if not chunks:
        raise ValueError("no chunks")
    if len(chunks) == 1 or overlap <= 0:
        return np.concatenate(chunks, axis=0)
    out = chunks[0].astype(np.float32)
    for nxt in chunks[1:]:
        nxt_f = nxt.astype(np.float32)
        ov = min(overlap, out.shape[0], nxt_f.shape[0])
        if ov > 0:
            for i in range(ov):
                a = (i + 1) / (ov + 1)
                out[-ov + i] = (1 - a) * out[-ov + i] + a * nxt_f[i]
            out = np.concatenate([out, nxt_f[ov:]], axis=0)
        else:
            out = np.concatenate([out, nxt_f], axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


@torch.no_grad()
def generate_research_video(
    prompt: str,
    *,
    steps: int | None = None,
    seed: int = 0,
    frames: int | None = None,
    fps: int | None = None,
    duration_sec: float | None = None,
    height: int | None = None,
    width: int | None = None,
    cfg_scale: float | None = None,
    out_path: Path | None = None,
    log_fn=None,
) -> str:
    cfg = LabConfig()
    device = get_device()
    steps = steps or cfg.sample_steps
    fps = max(4, int(fps or cfg.fps))
    height = height or cfg.height
    width = width or cfg.width
    cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale

    if duration_sec is not None:
        duration_sec = float(max(0.5, min(10.0, duration_sec)))
        frames = int(round(duration_sec * fps))
    elif frames is None:
        frames = int(getattr(cfg, "frames", 24))

    height = max(32, height - (height % 8))
    width = max(32, width - (width % 8))
    frames = max(8, frames - (frames % 4))
    max_frames = int(10 * fps)
    max_frames = max(8, max_frames - (max_frames % 4))
    if frames > max_frames:
        if log_fn:
            log_fn(f"Capping frames {frames} -> {max_frames} (10s @ {fps} fps)")
        frames = max_frames

    cfg.samples_dir.mkdir(parents=True, exist_ok=True)
    duration = frames / float(fps)
    if log_fn:
        log_fn(f"Generating {frames} frames @ {fps} fps ({duration:.2f}s), {width}x{height}")

    vae, dit, train_frames, train_h, train_w, _sc, _tc = _load_models(cfg, device, log_fn)
    chunk_f = max(8, train_frames - (train_frames % 4))
    chunk_h, chunk_w = train_h, train_w
    if log_fn:
        log_fn(
            f"Sampling from noise (no smoke prior). Train window {chunk_f}f @ {chunk_w}x{chunk_h}."
        )
        if frames > chunk_f or (height, width) != (chunk_h, chunk_w):
            log_fn(
                f"Chunked long-form: each chunk {chunk_f}f @ {chunk_w}x{chunk_h}, "
                f"then stitch/resize to {frames}f @ {width}x{height}"
            )

    vae.eval()
    dit.eval()

    n_chunks = max(1, int(np.ceil(frames / float(chunk_f))))
    chunks_u8: list[np.ndarray] = []
    for ci in range(n_chunks):
        if log_fn:
            log_fn(f"Chunk {ci + 1}/{n_chunks}")
        chunk = _sample_chunk_u8(
            vae=vae,
            dit=dit,
            prompt=prompt,
            chunk_frames=chunk_f,
            chunk_h=chunk_h,
            chunk_w=chunk_w,
            out_h=height,
            out_w=width,
            steps=steps,
            cfg_scale=float(cfg_scale),
            seed=seed,
            device=device,
            chunk_idx=ci,
        )
        chunks_u8.append(chunk)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frames_u8 = _crossfade_concat(chunks_u8, overlap=2)
    if frames_u8.shape[0] > frames:
        frames_u8 = frames_u8[:frames]
    elif frames_u8.shape[0] < frames:
        need = frames - frames_u8.shape[0]
        frames_u8 = np.concatenate([frames_u8, frames_u8[-need:]], axis=0)

    if log_fn:
        std = float(frames_u8.astype("float32").std())
        if std < 8:
            log_fn("Warning: output looks nearly flat — retrain VAE longer.")
        elif std > 90:
            log_fn("Warning: output still very noisy — train DiT longer / use in-domain prompts.")

    out_path = Path(out_path or (cfg.samples_dir / f"research_{seed}_{frames}f_{fps}fps.mp4"))
    write_rgb_video(frames_u8, out_path, fps=fps)
    if log_fn:
        log_fn(f"Wrote {out_path} ({frames_u8.shape[0] / float(fps):.2f}s)")
    return str(out_path)

from __future__ import annotations

import math

import torch
import torch.nn as nn

from video_lab.models.text_encoder import LocalTextEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def factored_rope_lite(t_len: int, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Factored 3D sinusoidal positional encoding (RoPE-lite): (T*H*W, dim).
    Not full rotary attention — additive spacetime PE for patch tokens.
    """
    assert dim % 6 == 0 or dim % 2 == 0
    # Split channels across t/h/w as evenly as possible
    d_each = max(2, (dim // 3) // 2 * 2)
    parts = []
    for length, offset_scale in ((t_len, 1.0), (h, 1.0), (w, 1.0)):
        half = d_each // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / max(half, 1))
        pos = torch.arange(length, device=device).float()
        args = pos[:, None] * freqs[None] * offset_scale
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # L, d_each
        parts.append(emb)
    # Broadcast to full grid
    t_emb = parts[0][:, None, None, :].expand(t_len, h, w, -1)
    h_emb = parts[1][None, :, None, :].expand(t_len, h, w, -1)
    w_emb = parts[2][None, None, :, :].expand(t_len, h, w, -1)
    pe = torch.cat([t_emb, h_emb, w_emb], dim=-1)  # T H W 3*d_each
    if pe.shape[-1] < dim:
        pe = torch.nn.functional.pad(pe, (0, dim - pe.shape[-1]))
    elif pe.shape[-1] > dim:
        pe = pe[..., :dim]
    return pe.reshape(t_len * h * w, dim)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm_c = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_tokens: int, hw: int) -> torch.Tensor:
        b, n, c = x.shape
        xs = x.view(b, t_tokens, hw, c).reshape(b * t_tokens, hw, c)
        h = self.norm1(xs)
        h, _ = self.attn_s(h, h, h, need_weights=False)
        xs = xs + h
        xs = xs.view(b, t_tokens, hw, c)
        xt = xs.permute(0, 2, 1, 3).reshape(b * hw, t_tokens, c)
        h = self.norm2(xt)
        h, _ = self.attn_t(h, h, h, need_weights=False)
        xt = xt + h
        x = xt.view(b, hw, t_tokens, c).permute(0, 2, 1, 3).reshape(b, n, c)
        h = self.norm_c(x)
        h, _ = self.cross(h, cond, cond, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm3(x))
        return x


class SpatioTemporalDiT(nn.Module):
    def __init__(
        self,
        latent_ch: int = 4,
        hidden: int = 192,
        layers: int = 4,
        heads: int = 4,
        text_dim: int = 192,
        max_text_len: int = 48,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.hidden = hidden
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.patch = nn.Conv3d(latent_ch, hidden, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.unpatch = nn.ConvTranspose3d(hidden, latent_ch, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.text_encoder = LocalTextEncoder(dim=text_dim, max_len=max_text_len)
        self.text_proj = nn.Linear(text_dim, hidden) if text_dim != hidden else nn.Identity()
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(hidden)

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        return self.text_proj(self.text_encoder(prompts))

    def forward(self, x: torch.Tensor, t: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        # x: B C T H W noisy latent
        b, c, tt, h, w = x.shape
        pt, ph, pw = self.patch_size
        # Pad latent so T/H/W divisible by patch
        pad_t = (pt - tt % pt) % pt
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        if pad_t or pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        tokens = self.patch(x)  # B hidden Tt Hh Ww
        _, _, tt2, h2, w2 = tokens.shape
        pe = factored_rope_lite(tt2, h2, w2, self.hidden, x.device)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(b, tt2 * h2 * w2, self.hidden)
        tokens = tokens + pe[None]
        temb = self.time_mlp(t)[:, None, :]
        tokens = tokens + temb
        cond = self.encode_text(prompts)
        for block in self.blocks:
            tokens = block(tokens, cond, tt2, h2 * w2)
        tokens = self.out_norm(tokens)
        tokens = tokens.view(b, tt2, h2, w2, self.hidden).permute(0, 4, 1, 2, 3)
        out = self.unpatch(tokens)
        return out[:, :, :tt, :h, :w]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """3D conv that does not look into the future (causal in time)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        self.pad_t = kt - 1
        self.pad_h = kh // 2
        self.pad_w = kw // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=(st, sh, sw), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, self.pad_t, 0))
        return self.conv(x)


class Encoder3D(nn.Module):
    """~8× spatial / 4× temporal compression with causal convs."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 4, base: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv3d(in_ch, base, (3, 3, 3), stride=(1, 2, 2)),  # T, H/2, W/2
            nn.SiLU(),
            CausalConv3d(base, base * 2, (3, 3, 3), stride=(2, 2, 2)),  # T/2, H/4, W/4
            nn.SiLU(),
            CausalConv3d(base * 2, base * 4, (3, 3, 3), stride=(2, 2, 2)),  # T/4, H/8, W/8
            nn.SiLU(),
            CausalConv3d(base * 4, latent_ch * 2, (3, 3, 3), stride=(1, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(self, out_ch: int = 3, latent_ch: int = 4, base: int = 48):
        super().__init__()
        self.in_conv = CausalConv3d(latent_ch, base * 4, (3, 3, 3))
        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up2 = nn.ConvTranspose3d(base * 2, base, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up3 = nn.ConvTranspose3d(base, base, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.out = CausalConv3d(base, out_ch, (3, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_conv(z))
        h = F.silu(self.up1(h))
        h = F.silu(self.up2(h))
        h = F.silu(self.up3(h))
        return torch.tanh(self.out(h))


class CausalVAE3D(nn.Module):
    def __init__(self, latent_ch: int = 4, base: int = 48):
        super().__init__()
        self.encoder = Encoder3D(latent_ch=latent_ch, base=base)
        self.decoder = Decoder3D(latent_ch=latent_ch, base=base)
        self.latent_ch = latent_ch
        self.base = base
        self.spatial_compress = 8
        self.temporal_compress = 4

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        logvar = logvar.clamp(-30, 20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
        return recon, mean, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x) + 0.5 * F.l1_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + 1e-5 * kl

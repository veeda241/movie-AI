"""Local trainable text encoder (no Hugging Face download)."""

from __future__ import annotations

import re

import torch
import torch.nn as nn


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalTextEncoder(nn.Module):
    """
    Word-piece-ish bag + small transformer — trained with the DiT.
    Better than char-hash: learns word meanings from your captions.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 8192,
        dim: int = 192,
        max_len: int = 48,
        layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(dim)

    def tokenize(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        batch: list[list[int]] = []
        for prompt in prompts:
            words = _TOKEN_RE.findall((prompt or "").lower())[: self.max_len]
            ids: list[int] = []
            for w in words:
                # Stable bucket into vocab (1..vocab-1); 0 = pad
                h = 2166136261
                for ch in w:
                    h ^= ord(ch)
                    h = (h * 16777619) & 0xFFFFFFFF
                ids.append((h % (self.vocab_size - 1)) + 1)
            if not ids:
                ids = [1]
            if len(ids) < self.max_len:
                ids = ids + [0] * (self.max_len - len(ids))
            batch.append(ids[: self.max_len])
        return torch.tensor(batch, device=device, dtype=torch.long)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        # Infer device from parameters
        device = next(self.parameters()).device
        ids = self.tokenize(prompts, device)
        b, n = ids.shape
        pos = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        key_padding = ids.eq(0)
        x = self.encoder(x, src_key_padding_mask=key_padding)
        return self.out_norm(x)

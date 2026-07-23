from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    device = device or get_device()
    if device.type == "cuda":
        return torch.float16
    return torch.float32

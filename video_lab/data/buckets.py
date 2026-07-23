"""Aspect-ratio training buckets with letterbox padding (no aggressive crop)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AspectBucket:
    name: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.name} ({self.width}x{self.height})"


# Lab-scale buckets (VRAM-safe). Train one bucket at a time in Phase A.
ASPECT_BUCKETS: tuple[AspectBucket, ...] = (
    AspectBucket("square_128", 128, 128),
    AspectBucket("square_256", 256, 256),
    AspectBucket("square_512", 512, 512),
    AspectBucket("landscape_384x256", 384, 256),
    AspectBucket("portrait_256x384", 256, 384),
    AspectBucket("square_96", 96, 96),
)

BUCKET_BY_NAME = {b.name: b for b in ASPECT_BUCKETS}


def list_bucket_labels() -> list[str]:
    return [b.label for b in ASPECT_BUCKETS]


def parse_bucket_choice(choice: str | None) -> AspectBucket:
    if not choice:
        return ASPECT_BUCKETS[0]
    choice = str(choice).strip()
    for b in ASPECT_BUCKETS:
        if choice == b.name or choice.startswith(b.name) or choice == b.label:
            return b
    # "128x128" style
    if "x" in choice.lower():
        try:
            w, h = choice.lower().replace(" ", "").split("x")[:2]
            w_i, h_i = int("".join(c for c in w if c.isdigit())), int("".join(c for c in h if c.isdigit()))
            for b in ASPECT_BUCKETS:
                if b.width == w_i and b.height == h_i:
                    return b
            return AspectBucket(f"custom_{w_i}x{h_i}", w_i, h_i)
        except Exception:
            pass
    return ASPECT_BUCKETS[0]


def choose_bucket_for_size(width: int, height: int) -> AspectBucket:
    """Pick closest bucket by aspect ratio, preferring smaller for lab VRAM."""
    if width <= 0 or height <= 0:
        return ASPECT_BUCKETS[0]
    ar = width / max(height, 1)
    best = ASPECT_BUCKETS[0]
    best_d = abs((best.width / best.height) - ar)
    for b in ASPECT_BUCKETS:
        # Prefer square_128 / square_96 for tiny sources
        d = abs((b.width / b.height) - ar)
        if d < best_d - 1e-6:
            best, best_d = b, d
        elif abs(d - best_d) < 1e-6 and b.width * b.height < best.width * best.height:
            best = b
    return best


def letterbox_frames(frames, target_h: int, target_w: int):
    """
    Letterbox/pad RGB float frames (T,H,W,3) in [0,1] to target size.
    Preserves aspect ratio — no aggressive center crop.
    """
    import numpy as np
    from PIL import Image

    t, h, w, c = frames.shape
    if h == target_h and w == target_w:
        return frames, {"pad_top": 0, "pad_left": 0, "scale": 1.0}

    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    out = np.zeros((t, target_h, target_w, c), dtype=np.float32)
    for i in range(t):
        pil = Image.fromarray((np.clip(frames[i], 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        out[i, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = arr
    meta = {"pad_top": pad_top, "pad_left": pad_left, "scale": float(scale), "content_h": new_h, "content_w": new_w}
    return out, meta

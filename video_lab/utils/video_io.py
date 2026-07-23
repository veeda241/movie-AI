from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def write_rgb_video(frames: np.ndarray, out_path: Path | str, fps: int = 8) -> str:
    import imageio_ffmpeg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3), got {frames.shape}")
    t, h, w, _ = frames.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 != h or w2 != w:
        frames = frames[:, :h2, :w2, :]
        h, w = h2, w2

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
    finally:
        code = proc.wait()
    if code != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {err}")
    return str(out_path)


def read_video_frames(
    path: Path | str,
    *,
    max_frames: int = 16,
    height: int = 64,
    width: int = 64,
    preserve_aspect: bool = False,
    max_side: int | None = None,
) -> np.ndarray:
    from PIL import Image

    path = Path(path)
    frames: list[np.ndarray] = []

    def _resize(img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img)
        if preserve_aspect:
            side = int(max_side or max(height, width))
            w0, h0 = pil.size
            scale = side / max(w0, h0, 1)
            nw, nh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
            pil = pil.resize((nw, nh), Image.BILINEAR)
        else:
            pil = pil.resize((width, height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0

    try:
        import av

        container = av.open(str(path))
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i >= max_frames:
                break
            img = frame.to_ndarray(format="rgb24")
            frames.append(_resize(img))
        container.close()
    except Exception:
        color = np.zeros((height, width, 3), dtype=np.float32)
        color[..., 0], color[..., 1], color[..., 2] = 0.2, 0.35, 0.55
        frames = [color.copy() for _ in range(max_frames)]

    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:max_frames], axis=0)


def tensor_from_frames(frames: np.ndarray):
    import torch

    x = torch.from_numpy(frames).float()
    x = x.permute(3, 0, 1, 2).unsqueeze(0)
    return x * 2.0 - 1.0


def frames_from_tensor(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
    if x.ndim == 5:
        x = x[0]
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).byte()
    return x.permute(1, 2, 3, 0).numpy()

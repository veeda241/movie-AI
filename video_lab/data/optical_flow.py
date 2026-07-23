"""Optical-flow motion metrics (OpenCV Farneback) for dataset filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FlowStats:
    mean_mag: float
    var_mag: float
    n_pairs: int

    def as_dict(self) -> dict:
        return {
            "flow_mean": round(self.mean_mag, 4),
            "flow_var": round(self.var_mag, 4),
            "flow_pairs": self.n_pairs,
        }


def compute_flow_stats(
    path: Path | str,
    *,
    sample_pairs: int = 10,
    resize: int = 96,
) -> FlowStats:
    """Return mean/variance of Farneback flow magnitude across sampled frame pairs."""
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return FlowStats(mean_mag=1.0, var_mag=0.0, n_pairs=0)

        mags: list[float] = []
        ok, prev = cap.read()
        if not ok:
            cap.release()
            return FlowStats(mean_mag=0.0, var_mag=0.0, n_pairs=0)
        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_g = cv2.resize(prev_g, (resize, resize))

        while len(mags) < sample_pairs:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize, resize))
            flow = cv2.calcOpticalFlowFarneback(
                prev_g,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(np.mean(mag)))
            prev_g = gray
        cap.release()
        if not mags:
            return FlowStats(mean_mag=0.0, var_mag=0.0, n_pairs=0)
        arr = np.asarray(mags, dtype=np.float64)
        return FlowStats(mean_mag=float(arr.mean()), var_mag=float(arr.var()), n_pairs=len(mags))
    except Exception:
        # If OpenCV missing / decode fails, do not block curation
        return FlowStats(mean_mag=1.0, var_mag=0.5, n_pairs=0)


def flow_filter_clip(
    path: Path | str,
    *,
    min_flow: float = 0.15,
    max_flow: float = 12.0,
    max_flow_var: float = 40.0,
    sample_pairs: int = 10,
) -> tuple[bool, FlowStats]:
    """
    Keep clips with moderate motion.
    - Near-zero mean → slideshow / static
    - Very high mean or variance → chaotic shake / glitch
    """
    stats = compute_flow_stats(path, sample_pairs=sample_pairs)
    if stats.n_pairs == 0 and stats.mean_mag == 0.0:
        return False, stats
    if stats.mean_mag < min_flow:
        return False, stats
    if stats.mean_mag > max_flow:
        return False, stats
    if stats.var_mag > max_flow_var:
        return False, stats
    return True, stats

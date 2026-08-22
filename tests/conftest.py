"""Shared fixtures for movie_pipeline tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Keys we touch in tests — saved/restored around each test.
_ENV_KEYS = [
    "MINIMAX_API_KEY",
    "MINIMAX_API_BASE",
    "MINIMAX_DURATION",
    "MINIMAX_RESOLUTION",
    "MINIMAX_RATIO",
    "MINIMAX_POLL_INTERVAL",
    "MINIMAX_MAX_POLL_ATTEMPTS",
    "MINIMAX_REQUEST_TIMEOUT",
    "HF_TOKEN",
    "HF_ALLOW_LOCAL_FALLBACK",
    "HF_FORCE_LOCAL_VIDEO",
    "HF_VIDEO_PROVIDER",
    "HF_VIDEO_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(tmp_path: Path) -> None:
    """Snapshot and restore env vars around every test, clearing MiniMax/HF keys."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    # Clear all relevant keys
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    yield
    # Restore original values
    for k in _ENV_KEYS:
        if saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for generated files."""
    d = tmp_path / "output"
    d.mkdir()
    return d

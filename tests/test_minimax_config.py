"""Tests for MiniMaxH3Config defaults, from_env, and edge cases."""

from __future__ import annotations

import os

import pytest

from movie_pipeline.models.generative_config import (
    MiniMaxH3Config,
    ModelFamily,
    VideoGenerationConfig,
)


# ── ModelFamily enum ─────────────────────────────────────────────────


class TestModelFamily:
    def test_minimax_h3_member(self) -> None:
        assert ModelFamily.MINIMAX_H3 == "minimax-h3"

    def test_all_members_are_strings(self) -> None:
        for member in ModelFamily:
            assert isinstance(member, str)


# ── MiniMaxH3Config defaults ─────────────────────────────────────────


class TestMiniMaxH3ConfigDefaults:
    def test_model_name(self) -> None:
        assert MiniMaxH3Config().model_name == "MiniMax-H3"

    def test_base_url(self) -> None:
        assert MiniMaxH3Config().base_url == "https://api.minimax.io"

    def test_duration_default(self) -> None:
        assert MiniMaxH3Config().duration == 5

    def test_resolution_default(self) -> None:
        assert MiniMaxH3Config().resolution == "768P"

    def test_ratio_default(self) -> None:
        assert MiniMaxH3Config().ratio == "16:9"

    def test_poll_interval_default(self) -> None:
        assert MiniMaxH3Config().poll_interval == 10

    def test_max_poll_attempts_default(self) -> None:
        assert MiniMaxH3Config().max_poll_attempts == 60

    def test_request_timeout_default(self) -> None:
        assert MiniMaxH3Config().request_timeout == 30

    def test_frozen_dataclass(self) -> None:
        cfg = MiniMaxH3Config()
        with pytest.raises(AttributeError):
            cfg.duration = 10  # type: ignore[misc]


# ── MiniMaxH3Config.from_env ─────────────────────────────────────────


class TestMiniMaxH3ConfigFromEnv:
    def test_all_defaults_when_no_env(self) -> None:
        cfg = MiniMaxH3Config.from_env()
        assert cfg.base_url == "https://api.minimax.io"
        assert cfg.duration == 5
        assert cfg.resolution == "768P"
        assert cfg.ratio == "16:9"
        assert cfg.poll_interval == 10
        assert cfg.max_poll_attempts == 60
        assert cfg.request_timeout == 30

    def test_custom_base_url(self) -> None:
        os.environ["MINIMAX_API_BASE"] = "https://custom.api.io"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.base_url == "https://custom.api.io"

    def test_custom_duration(self) -> None:
        os.environ["MINIMAX_DURATION"] = "15"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.duration == 15

    def test_custom_resolution(self) -> None:
        os.environ["MINIMAX_RESOLUTION"] = "2K"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.resolution == "2K"

    def test_custom_ratio(self) -> None:
        os.environ["MINIMAX_RATIO"] = "9:16"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.ratio == "9:16"

    def test_custom_poll_interval(self) -> None:
        os.environ["MINIMAX_POLL_INTERVAL"] = "5"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.poll_interval == 5

    def test_custom_max_poll_attempts(self) -> None:
        os.environ["MINIMAX_MAX_POLL_ATTEMPTS"] = "3"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.max_poll_attempts == 3

    def test_custom_request_timeout(self) -> None:
        os.environ["MINIMAX_REQUEST_TIMEOUT"] = "60"
        cfg = MiniMaxH3Config.from_env()
        assert cfg.request_timeout == 60

    def test_invalid_duration_raises(self) -> None:
        os.environ["MINIMAX_DURATION"] = "not-a-number"
        with pytest.raises(ValueError):
            MiniMaxH3Config.from_env()

    def test_invalid_poll_interval_raises(self) -> None:
        os.environ["MINIMAX_POLL_INTERVAL"] = "abc"
        with pytest.raises(ValueError):
            MiniMaxH3Config.from_env()


# ── VideoGenerationConfig MINIMAX auto-detect ────────────────────────


class TestVideoGenerationConfigMinimaxDetection:
    def test_defaults_to_motif_video_without_key(self) -> None:
        cfg = VideoGenerationConfig.from_env()
        assert cfg.family == ModelFamily.MOTIF_VIDEO
        assert cfg.provider == "hf-inference"

    def test_switches_to_minimax_when_key_set(self) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key-123"
        cfg = VideoGenerationConfig.from_env()
        assert cfg.family == ModelFamily.MINIMAX_H3
        assert cfg.model_id == "MiniMax-H3"
        assert cfg.provider == "minimax"

    def test_whitespace_key_treated_as_empty(self) -> None:
        os.environ["MINIMAX_API_KEY"] = "   "
        cfg = VideoGenerationConfig.from_env()
        assert cfg.family == ModelFamily.MOTIF_VIDEO

    def test_env_override_takes_precedence(self) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        os.environ["VIDEO_MODEL_FAMILY"] = "flux"
        os.environ["HF_VIDEO_MODEL"] = "custom-model"
        os.environ["HF_VIDEO_PROVIDER"] = "custom-provider"
        cfg = VideoGenerationConfig.from_env()
        assert cfg.family.value == "flux"
        assert cfg.model_id == "custom-model"
        assert cfg.provider == "custom-provider"

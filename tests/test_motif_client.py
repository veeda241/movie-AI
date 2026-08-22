"""Tests for MotifClient routing: MiniMax H3, HF Inference, and local fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from movie_pipeline.video.motif_client import MotifClient, UI_MODEL_PRESETS


# ── UI_MODEL_PRESETS ─────────────────────────────────────────────────


class TestUIModelPresets:
    def test_wan_22_preset(self) -> None:
        preset = UI_MODEL_PRESETS["wan-2.2"]
        assert preset["provider"] == "fal-ai"
        assert preset["model_id"] == "Wan-AI/Wan2.2-T2V-A14B"

    def test_minimax_h3_preset(self) -> None:
        preset = UI_MODEL_PRESETS["minimax-h3"]
        assert preset["provider"] == "minimax"
        assert preset["model_id"] == "MiniMax-H3"


# ── MotifClient constructor routing ──────────────────────────────────


class TestMotifClientInit:
    def test_default_provider_without_key(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.remote_provider == "hf-inference"
        assert mfc.remote_model == "ali-vilab/text-to-video-ms-1.7b"
        assert mfc.minimax_client is None

    def test_minimax_client_created_when_key_and_provider(self, output_dir: Path) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.remote_provider == "minimax"
        assert mfc.minimax_client is not None

    def test_minimax_client_not_created_when_no_key(self, output_dir: Path) -> None:
        os.environ.pop("MINIMAX_API_KEY", None)
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.minimax_client is None

    def test_minimax_client_not_created_when_provider_not_minimax(
        self, output_dir: Path
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        mfc = MotifClient(output_dir=output_dir, provider="hf-inference")
        assert mfc.minimax_client is None

    def test_wan_22_ui_model_overrides_provider(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir, ui_model="wan-2.2")
        assert mfc.remote_provider == "fal-ai"
        assert mfc.remote_model == "Wan-AI/Wan2.2-T2V-A14B"

    def test_minimax_h3_ui_model_overrides_provider(self, output_dir: Path) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        mfc = MotifClient(output_dir=output_dir, ui_model="minimax-h3")
        assert mfc.remote_provider == "minimax"
        assert mfc.remote_model == "MiniMax-H3"
        assert mfc.minimax_client is not None

    def test_explicit_provider_overrides_preset(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir, provider="custom-provider", ui_model="wan-2.2")
        assert mfc.remote_provider == "custom-provider"

    def test_explicit_model_overrides_preset(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir, model_id="custom/model")
        assert mfc.remote_model == "custom/model"

    def test_force_local_flag(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir, force_local=True)
        assert mfc.force_local is True

    def test_force_local_from_env(self, output_dir: Path) -> None:
        os.environ["HF_FORCE_LOCAL_VIDEO"] = "true"
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.force_local is True

    def test_back_compat_aliases(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.REMOTE_PROVIDER == mfc.remote_provider
        assert mfc.REMOTE_MODEL == mfc.remote_model


# ── MotifClient.generate() routing ───────────────────────────────────


class TestMotifClientGenerateRouting:
    def test_force_local_generates_placeholder(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mfc = MotifClient(output_dir=output_dir, force_local=True)
        result = mfc.generate("A cat in a hallway", 42)
        assert result
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0
        Path(result).unlink(missing_ok=True)

    def test_minimax_client_called_when_available(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.minimax_client is not None

        # Mock the MiniMax client's generate method
        fake_result = str(output_dir / "minimax_video.mp4")
        mfc.minimax_client.generate = MagicMock(return_value=fake_result)

        result = mfc.generate("A dramatic scene", 7)
        assert result == fake_result
        mfc.minimax_client.generate.assert_called_once_with(
            "A dramatic scene", 7, output_path=output_dir / "scene_7_video.mp4",
            first_frame_image=None,
        )

    def test_minimax_failure_falls_back_to_local(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        os.environ["HF_ALLOW_LOCAL_FALLBACK"] = "true"
        mfc = MotifClient(output_dir=output_dir)

        # Make MiniMax fail
        mfc.minimax_client.generate = MagicMock(
            side_effect=RuntimeError("MiniMax API down")
        )

        result = mfc.generate("A dramatic scene", 7)
        # Should fall through to local video
        assert result
        assert Path(result).exists()
        Path(result).unlink(missing_ok=True)

    def test_minimax_failure_no_fallback_raises(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        os.environ["HF_ALLOW_LOCAL_FALLBACK"] = "false"
        mfc = MotifClient(output_dir=output_dir)

        mfc.minimax_client.generate = MagicMock(
            side_effect=RuntimeError("MiniMax API down")
        )

        with pytest.raises(RuntimeError, match="MiniMax H3 failed.*ALLOW_LOCAL_FALLBACK=false"):
            mfc.generate("A dramatic scene", 7)

    def test_no_token_no_minimax_goes_local(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("MINIMAX_API_KEY", None)
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.token == ""
        assert mfc.minimax_client is None

        result = mfc.generate("test", 1)
        assert result
        assert Path(result).exists()
        Path(result).unlink(missing_ok=True)

    def test_custom_output_path(self, output_dir: Path) -> None:
        mfc = MotifClient(output_dir=output_dir, force_local=True)
        custom = output_dir / "custom_dir" / "my_video.mp4"
        result = mfc.generate("test", 1, output_path=custom)
        assert result == str(custom)
        assert custom.exists()


# ── _local_fallback_allowed() ────────────────────────────────────────


class TestMotifClientImageToVideo:
    def test_first_frame_passed_to_minimax(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "test-key"
        mfc = MotifClient(output_dir=output_dir)
        assert mfc.minimax_client is not None

        fake_image = output_dir / "keyframe.png"
        fake_image.write_bytes(b"fake-png")
        fake_result = str(output_dir / "video.mp4")
        mfc.minimax_client.generate = MagicMock(return_value=fake_result)

        mfc.generate("Dance", 3, first_frame_image=fake_image)

        mfc.minimax_client.generate.assert_called_once_with(
            "Dance", 3, output_path=output_dir / "scene_3_video.mp4",
            first_frame_image=fake_image,
        )

    def test_first_frame_none_when_no_keyframe(self, output_dir: Path) -> None:
        os.environ.pop("MINIMAX_API_KEY", None)
        mfc = MotifClient(output_dir=output_dir, force_local=True)
        # Should not crash — local path ignores first_frame_image
        result = mfc.generate("test", 1)
        assert result
        Path(result).unlink(missing_ok=True)


class TestLocalFallbackAllowed:
    def test_default_is_true(self) -> None:
        from movie_pipeline.video.motif_client import _local_fallback_allowed

        assert _local_fallback_allowed() is True

    def test_false_when_disabled(self) -> None:
        os.environ["HF_ALLOW_LOCAL_FALLBACK"] = "false"
        from movie_pipeline.video.motif_client import _local_fallback_allowed

        assert _local_fallback_allowed() is False

    def test_false_when_off(self) -> None:
        os.environ["HF_ALLOW_LOCAL_FALLBACK"] = "off"
        from movie_pipeline.video.motif_client import _local_fallback_allowed

        assert _local_fallback_allowed() is False

    def test_true_when_enabled(self) -> None:
        os.environ["HF_ALLOW_LOCAL_FALLBACK"] = "true"
        from movie_pipeline.video.motif_client import _local_fallback_allowed

        assert _local_fallback_allowed() is True

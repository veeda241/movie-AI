"""Tests for MiniMaxClient with mocked HTTP requests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from movie_pipeline.models.generative_config import MiniMaxH3Config
from movie_pipeline.video.minimax_client import MiniMaxClient


# ── Helpers ──────────────────────────────────────────────────────────


def _make_client(tmp_path: Path, *, api_key: str = "test-key") -> MiniMaxClient:
    os.environ["MINIMAX_API_KEY"] = api_key
    cfg = MiniMaxH3Config(
        poll_interval=0,
        max_poll_attempts=3,
        request_timeout=5,
        output_dir=tmp_path,
    )
    return MiniMaxClient(output_dir=tmp_path, config=cfg)


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    content: bytes = b"fake-video-bytes",
    *,
    raise_for_status: bool = False,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.iter_content.return_value = [content]
    resp.raise_for_status.side_effect = (
        Exception(f"HTTP {status_code}") if raise_for_status else None
    )
    return resp


# ── Constructor tests ────────────────────────────────────────────────


class TestMiniMaxClientConstructor:
    def test_stores_api_key(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, api_key="sk-abc123")
        assert client.api_key == "sk-abc123"

    def test_strips_whitespace_from_key(self, tmp_path: Path) -> None:
        os.environ["MINIMAX_API_KEY"] = "  sk-abc  "
        cfg = MiniMaxH3Config(poll_interval=0, max_poll_attempts=1, output_dir=tmp_path)
        client = MiniMaxClient(output_dir=tmp_path, config=cfg)
        assert client.api_key == "sk-abc"

    def test_empty_key_when_not_set(self, tmp_path: Path) -> None:
        os.environ.pop("MINIMAX_API_KEY", None)
        cfg = MiniMaxH3Config(poll_interval=0, max_poll_attempts=1, output_dir=tmp_path)
        client = MiniMaxClient(output_dir=tmp_path, config=cfg)
        assert client.api_key == ""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "new_dir"
        assert not target.exists()
        _make_client(target)
        assert target.exists()

    def test_config_applied(self, tmp_path: Path) -> None:
        os.environ["MINIMAX_API_KEY"] = "key"
        cfg = MiniMaxH3Config(duration=10, resolution="2K", ratio="9:16", output_dir=tmp_path)
        client = MiniMaxClient(output_dir=tmp_path, config=cfg)
        assert client.cfg.duration == 10
        assert client.cfg.resolution == "2K"
        assert client.cfg.ratio == "9:16"


# ── generate() — no API key ─────────────────────────────────────────


class TestMiniMaxClientNoApiKey:
    def test_raises_runtime_error(self, tmp_path: Path) -> None:
        os.environ.pop("MINIMAX_API_KEY", None)
        cfg = MiniMaxH3Config(output_dir=tmp_path)
        client = MiniMaxClient(output_dir=tmp_path, config=cfg)
        with pytest.raises(RuntimeError, match="MINIMAX_API_KEY is not set"):
            client.generate("test prompt", 1)


# ── generate() — full success flow ───────────────────────────────────


class TestMiniMaxClientSuccessFlow:
    def test_creates_task_polls_and_downloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _make_client(tmp_path)

        # Mock responses for: create task → poll (pending) → poll (succeeded) → download
        task_response = _mock_response(json_data={"task_id": "task-123"})
        pending_response = _mock_response(json_data={"task": {"status": "pending"}})
        success_response = _mock_response(
            json_data={
                "task": {
                    "status": "succeeded",
                    "content": {"url": "https://cdn.example.com/video.mp4"},
                }
            }
        )
        download_response = _mock_response(content=b"real-video-data-here")

        responses = [task_response, pending_response, success_response, download_response]
        call_count = {"n": 0}

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            r = responses[call_count["n"]]
            call_count["n"] += 1
            return r

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            r = responses[call_count["n"]]
            call_count["n"] += 1
            return r

        # Patch time.sleep to avoid waiting, and session methods
        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        result = client.generate("A cat walking", 5)

        assert Path(result).exists()
        assert Path(result).read_bytes() == b"real-video-data-here"
        assert "scene_5_video.mp4" in result

    def test_custom_output_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)

        task_response = _mock_response(json_data={"task_id": "task-456"})
        success_response = _mock_response(
            json_data={
                "task": {
                    "status": "succeeded",
                    "content": {"url": "https://cdn.example.com/v.mp4"},
                }
            }
        )
        download_response = _mock_response(content=b"data")

        responses = [task_response, success_response, download_response]
        call_count = {"n": 0}

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            r = responses[call_count["n"]]
            call_count["n"] += 1
            return r

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            r = responses[call_count["n"]]
            call_count["n"] += 1
            return r

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        custom = tmp_path / "custom" / "my_video.mp4"
        result = client.generate("prompt", 1, output_path=custom)
        assert result == str(custom)
        assert custom.exists()


# ── generate() — API errors ──────────────────────────────────────────


class TestMiniMaxClientErrors:
    def test_task_creation_no_task_id_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        client._session.post = MagicMock(
            return_value=_mock_response(json_data={"error": "bad request"})
        )
        with pytest.raises(RuntimeError, match="did not return a task_id"):
            client.generate("prompt", 1)

    def test_task_creation_http_error_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        resp = _mock_response(status_code=401, raise_for_status=True)
        client._session.post = MagicMock(return_value=resp)
        with pytest.raises(Exception, match="HTTP 401"):
            client.generate("prompt", 1)

    def test_polling_max_attempts_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        task_resp = _mock_response(json_data={"task_id": "t1"})
        pending_resp = _mock_response(json_data={"task": {"status": "pending"}})

        post_calls = {"n": 0}
        get_calls = {"n": 0}

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            post_calls["n"] += 1
            return task_resp

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            get_calls["n"] += 1
            return pending_resp

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        with pytest.raises(RuntimeError, match="did not finish after"):
            client.generate("prompt", 1)

    def test_polling_task_failed_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        task_resp = _mock_response(json_data={"task_id": "t2"})
        fail_resp = _mock_response(
            json_data={"task": {"status": "failed", "error": {"message": "NSFW content"}}}
        )

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            return task_resp

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            return fail_resp

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        with pytest.raises(RuntimeError, match="status=failed"):
            client.generate("prompt", 1)

    def test_polling_task_cancelled_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        task_resp = _mock_response(json_data={"task_id": "t3"})
        cancel_resp = _mock_response(
            json_data={"task": {"status": "cancelled"}}
        )

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            return task_resp

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            return cancel_resp

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        with pytest.raises(RuntimeError, match="status=cancelled"):
            client.generate("prompt", 1)

    def test_succeeded_without_url_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)
        task_resp = _mock_response(json_data={"task_id": "t4"})
        success_resp = _mock_response(
            json_data={"task": {"status": "succeeded", "content": {}}}
        )

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            return task_resp

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            return success_resp

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        with pytest.raises(RuntimeError, match="content\\.url is missing"):
            client.generate("prompt", 1)


# ── Payload construction ─────────────────────────────────────────────


class TestMiniMaxClientPayload:
    def test_create_task_payload_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        os.environ["MINIMAX_API_KEY"] = "sk-test"
        cfg = MiniMaxH3Config(duration=10, resolution="2K", ratio="9:16", output_dir=tmp_path)
        client = MiniMaxClient(output_dir=tmp_path, config=cfg)

        captured_kwargs: dict[str, Any] = {}

        def mock_post(url: str, **kwargs: Any) -> MagicMock:
            captured_kwargs["url"] = url
            captured_kwargs.update(kwargs)
            return _mock_response(json_data={"task_id": "t-payload"})

        client._session.post = mock_post

        # Mock poll to immediately succeed
        client._session.get = MagicMock(
            return_value=_mock_response(
                json_data={
                    "task": {
                        "status": "succeeded",
                        "content": {"url": "https://example.com/v.mp4"},
                    }
                }
            )
        )
        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)

        # Need a download mock too
        original_get = client._session.get

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            if args and "query" in str(args[0]):
                return _mock_response(
                    json_data={
                        "task": {
                            "status": "succeeded",
                            "content": {"url": "https://example.com/v.mp4"},
                        }
                    }
                )
            return _mock_response(content=b"video-bytes")

        client._session.get = mock_get

        client.generate("A dramatic sunset", 3, duration=10, resolution="2K", ratio="9:16")

        assert captured_kwargs["url"] == "https://api.minimax.io/v2/video_generation"
        payload = captured_kwargs["json"]
        assert payload["model"] == "MiniMax-H3"
        assert payload["duration"] == 10
        assert payload["resolution"] == "2K"
        assert payload["ratio"] == "9:16"
        assert payload["content"] == [{"type": "text", "text": "A dramatic sunset"}]


# ── Image-to-video (first_frame) ─────────────────────────────────────


class TestMiniMaxClientImageToVideo:
    def test_payload_includes_first_frame_when_image_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _make_client(tmp_path)

        # Create a fake PNG image
        image_path = tmp_path / "keyframe.png"
        image_path.write_bytes(b"\x89PNGfake-image-data")

        captured_payload: dict[str, Any] = {}

        def mock_post(url: str, **kwargs: Any) -> MagicMock:
            captured_payload.update(kwargs)
            return _mock_response(json_data={"task_id": "t-img"})

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            return _mock_response(
                json_data={
                    "task": {
                        "status": "succeeded",
                        "content": {"url": "https://example.com/v.mp4"},
                    }
                }
            )

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        client.generate("Dance scene", 1, first_frame_image=image_path)

        content = captured_payload["json"]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Dance scene"}
        assert content[1]["type"] == "image_url"
        assert content[1]["role"] == "first_frame"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_payload_text_only_when_no_image(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client(tmp_path)

        captured_payload: dict[str, Any] = {}

        def mock_post(url: str, **kwargs: Any) -> MagicMock:
            captured_payload.update(kwargs)
            return _mock_response(json_data={"task_id": "t-no-img"})

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            return _mock_response(
                json_data={
                    "task": {
                        "status": "succeeded",
                        "content": {"url": "https://example.com/v.mp4"},
                    }
                }
            )

        monkeypatch.setattr("movie_pipeline.video.minimax_client.time.sleep", lambda _: None)
        client._session.post = mock_post
        client._session.get = mock_get

        client.generate("Sunset scene", 1)

        content = captured_payload["json"]["content"]
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Sunset scene"}

    def test_missing_image_file_raises(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(FileNotFoundError, match="first_frame_image not found"):
            client._create_task(
                "prompt", duration=5, resolution="768P", ratio="16:9",
                first_frame_image=missing,
            )

    def test_data_uri_encoding_jpg(self, tmp_path: Path) -> None:
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        uri = MiniMaxClient._encode_image_as_data_uri(img)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_data_uri_encoding_webp(self, tmp_path: Path) -> None:
        img = tmp_path / "frame.webp"
        img.write_bytes(b"RIFFfake-webp")
        uri = MiniMaxClient._encode_image_as_data_uri(img)
        assert uri.startswith("data:image/webp;base64,")

    def test_data_uri_encoding_fallback_suffix(self, tmp_path: Path) -> None:
        img = tmp_path / "frame.tiff"
        img.write_bytes(b"\x4d\x4d\x00\x2afake-tiff")
        uri = MiniMaxClient._encode_image_as_data_uri(img)
        assert uri.startswith("data:image/tiff;base64,")

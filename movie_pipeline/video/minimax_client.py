"""MiniMax H3 video generation client.

Workflow (3-step async):
    1. POST /v2/video_generation  →  submit task, receive task_id
    2. GET  /v2/query/video_generation/{task_id}  →  poll until succeeded/failed
    3. Download the video from content.url
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from movie_pipeline.models.generative_config import MiniMaxH3Config


class MiniMaxClient:
    """Synchronous MiniMax H3 text-to-video client with polling + download."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        *,
        config: MiniMaxH3Config | None = None,
    ) -> None:
        self.cfg = config or MiniMaxH3Config.from_env()
        self.api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        self.output_dir = (
            Path(output_dir) if output_dir is not None else self.cfg.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        video_prompt: str,
        scene_number: int,
        *,
        output_path: str | Path | None = None,
        first_frame_image: str | Path | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        ratio: str | None = None,
    ) -> str:
        """Generate a video from *video_prompt* and return the local file path.

        When *first_frame_image* is provided the task runs in image-to-video
        mode — the image is uploaded as a ``first_frame`` content entry and
        the API uses it as the opening frame.  In this mode the *ratio*
        parameter is ignored (the API determines ratio from the image).

        Raises ``RuntimeError`` on terminal failure so callers can decide
        whether to fall back.
        """
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not set. "
                "Get an API key at https://platform.minimax.io"
            )

        target = (
            Path(output_path)
            if output_path is not None
            else self.output_dir / f"scene_{scene_number}_video.mp4"
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        duration = duration or self.cfg.duration
        resolution = resolution or self.cfg.resolution
        ratio = ratio or self.cfg.ratio

        task_id = self._create_task(
            video_prompt,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            first_frame_image=first_frame_image,
        )
        print(
            f"[MiniMax] scene {scene_number}: task {task_id} submitted",
            flush=True,
        )

        video_url = self._poll_task(task_id, scene_number=scene_number)
        print(
            f"[MiniMax] scene {scene_number}: downloading from {video_url}",
            flush=True,
        )

        self._download(video_url, target)
        print(
            f"[MiniMax] scene {scene_number}: saved {target}",
            flush=True,
        )
        return str(target)

    # ------------------------------------------------------------------
    # Step 1 – Create task
    # ------------------------------------------------------------------

    def _create_task(
        self,
        prompt: str,
        *,
        duration: int,
        resolution: str,
        ratio: str,
        first_frame_image: str | Path | None = None,
    ) -> str:
        url = f"{self.cfg.base_url}/v2/video_generation"

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if first_frame_image is not None:
            image_path = Path(first_frame_image)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"first_frame_image not found: {image_path}"
                )
            image_data_uri = self._encode_image_as_data_uri(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data_uri},
                "role": "first_frame",
            })

        payload: dict[str, Any] = {
            "model": self.cfg.model_name,
            "content": content,
            "duration": duration,
            "resolution": resolution,
            "ratio": ratio,
        }

        resp = self._session.post(
            url,
            headers=self._headers,
            json=payload,
            timeout=self.cfg.request_timeout,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(
                f"MiniMax API did not return a task_id: {data}"
            )
        return str(task_id)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_as_data_uri(image_path: Path) -> str:
        """Read an image file and return a ``data:image/...;base64,...`` URI.

        The MiniMax API accepts both data URIs and HTTP URLs for image
        inputs.  Data URIs work offline without requiring the image to be
        publicly hosted.
        """
        import base64

        suffix = image_path.suffix.lower().lstrip(".") or "png"
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "heic": "image/heic",
            "heif": "image/heif",
        }
        mime = mime_map.get(suffix, f"image/{suffix}")
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # ------------------------------------------------------------------
    # Step 2 – Poll until terminal status
    # ------------------------------------------------------------------

    def _poll_task(self, task_id: str, *, scene_number: int = 0) -> str:
        """Poll until *succeeded* / *failed* / *cancelled* and return the video URL."""
        url = f"{self.cfg.base_url}/v2/query/video_generation/{task_id}"

        for attempt in range(1, self.cfg.max_poll_attempts + 1):
            time.sleep(self.cfg.poll_interval)
            resp = self._session.get(url, headers=self._headers, timeout=self.cfg.request_timeout)
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
            task: dict[str, Any] = body.get("task", body)
            status = str(task.get("status", "")).lower()
            print(
                f"[MiniMax] scene {scene_number}: poll {attempt}/{self.cfg.max_poll_attempts} — {status}",
                flush=True,
            )

            if status == "succeeded":
                content: dict[str, Any] = task.get("content", {})
                video_url = content.get("url")
                if not video_url:
                    raise RuntimeError(
                        f"Task succeeded but content.url is missing: {task}"
                    )
                return str(video_url)

            if status in ("failed", "cancelled"):
                error_info = task.get("error", {})
                raise RuntimeError(
                    f"MiniMax task {task_id} ended with status={status}: "
                    f"{error_info}"
                )

        raise RuntimeError(
            f"MiniMax task {task_id} did not finish after "
            f"{self.cfg.max_poll_attempts} polls ({self.cfg.poll_interval}s interval)"
        )

    # ------------------------------------------------------------------
    # Step 3 – Download
    # ------------------------------------------------------------------

    def _download(self, url: str, target: Path) -> None:
        resp = self._session.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                f.write(chunk)

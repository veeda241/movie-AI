from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests


class MiniMaxH3Client:
    """Client for the MiniMax H3 video generation API.

    Environment variables:
        MINIMAX_API_KEY  – API token from platform.minimax.io
        MINIMAX_API_BASE – Override the API base URL
                        (default https://api.minimax.io)
    """

    DEFAULT_BASE_URL = "https://api.minimax.io"
    MODEL_NAME = "MiniMax-H3"
    DEFAULT_DURATION = int(os.environ.get("MINIMAX_DURATION", "5"))
    DEFAULT_RESOLUTION = os.environ.get("MINIMAX_RESOLUTION", "768P")
    DEFAULT_RATIO = os.environ.get("MINIMAX_RATIO", "16:9")
    POLL_INTERVAL_SECONDS = int(os.environ.get("MINIMAX_POLL_INTERVAL", "10"))
    MAX_POLL_ATTEMPTS = int(os.environ.get("MINIMAX_MAX_POLL_ATTEMPTS", "60"))
    REQUEST_TIMEOUT = int(os.environ.get("MINIMAX_REQUEST_TIMEOUT", "30"))

    def __init__(self) -> None:
        self.api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        self.base_url = os.environ.get("MINIMAX_API_BASE", self.DEFAULT_BASE_URL).rstrip("/")
        self.output_dir = Path(__file__).resolve().parents[1] / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, video_prompt: str, scene_number: int) -> str:
        """Generate a video using the MiniMax H3 API.

        Returns the local file path of the downloaded video, or empty string on failure.
        """
        if not self.available:
            return ""

        try:
            task_id = self._create_task(video_prompt)
            print(f"[MiniMaxH3] scene {scene_number}: task submitted (id={task_id})", flush=True)

            download_url = self._poll_task(task_id, scene_number)
            if not download_url:
                return ""

            return self._download_video(download_url, scene_number)
        except Exception as exc:
            print(f"[MiniMaxH3] scene {scene_number} failed: {exc}", flush=True)
            return ""

    # ------------------------------------------------------------------
    # Step 1 – Create a video generation task
    # ------------------------------------------------------------------
    def _create_task(self, prompt: str) -> str:
        url = f"{self.base_url}/v2/video_generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.MODEL_NAME,
            "content": [
                {"type": "text", "text": prompt},
            ],
            "duration": self.DEFAULT_DURATION,
            "resolution": self.DEFAULT_RESOLUTION,
            "ratio": self.DEFAULT_RATIO,
        }
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        body = response.json()

        if "task_id" not in body:
            raise RuntimeError(f"MiniMax API did not return a task_id: {body}")

        return str(body["task_id"])

    # ------------------------------------------------------------------
    # Step 2 – Poll task status until succeeded / failed / cancelled
    # ------------------------------------------------------------------
    def _poll_task(self, task_id: str, scene_number: int) -> str:
        url = f"{self.base_url}/v2/query/video_generation/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(1, self.MAX_POLL_ATTEMPTS + 1):
            time.sleep(self.POLL_INTERVAL_SECONDS)

            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            body = response.json()

            task = body.get("task", body)
            status = task.get("status", "unknown")

            if status == "succeeded":
                video_url = task.get("content", {}).get("url", "")
                if not video_url:
                    raise RuntimeError(f"Task succeeded but no URL in response: {task}")
                print(
                    f"[MiniMaxH3] scene {scene_number}: generation succeeded "
                    f"({attempt} polls)",
                    flush=True,
                )
                return video_url

            if status in ("failed", "cancelled"):
                error_info = task.get("error", {})
                raise RuntimeError(
                    f"Task {status}: {error_info or 'no error details'}"
                )

            print(
                f"[MiniMaxH3] scene {scene_number}: status={status} "
                f"(poll {attempt}/{self.MAX_POLL_ATTEMPTS})",
                flush=True,
            )

        raise RuntimeError(
            f"Task did not finish within {self.MAX_POLL_ATTEMPTS} polls "
            f"({self.MAX_POLL_ATTEMPTS * self.POLL_INTERVAL_SECONDS}s)"
        )

    # ------------------------------------------------------------------
    # Step 3 – Download the generated video
    # ------------------------------------------------------------------
    def _download_video(self, download_url: str, scene_number: int) -> str:
        out_path = self.output_dir / f"scene_{scene_number}_video.mp4"

        response = requests.get(download_url, timeout=120)
        response.raise_for_status()
        out_path.write_bytes(response.content)

        size_mb = len(response.content) / (1024 * 1024)
        print(
            f"[MiniMaxH3] scene {scene_number}: downloaded {size_mb:.1f} MB -> {out_path}",
            flush=True,
        )
        return str(out_path)

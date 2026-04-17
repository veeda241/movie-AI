from __future__ import annotations

import json
from typing import Any

try:
    from movie_pipeline.agents.base import call_hf_json
except ImportError:  # pragma: no cover - direct execution fallback
    from agents.base import call_hf_json


class CinematographerAgent:
    def run(self, screenwriter_output: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(screenwriter_output, dict):
            screenwriter_output = screenwriter_output.get("screenwriter_output", [])
        if not isinstance(screenwriter_output, list):
            raise TypeError("CinematographerAgent expected a list of script blocks.")

        shot_lists: list[dict[str, Any]] = []
        for scene in screenwriter_output:
            if not isinstance(scene, dict):
                raise TypeError("Each screenwriter scene must be a dictionary.")

            shot_lists.append(self._run_scene(scene))

        return shot_lists

    def _run_scene(self, scene: dict[str, Any]) -> dict[str, Any]:
        scene_number = int(scene.get("scene_number", 0))
        prompt = (
            "You are a cinematographer. Given this single scene script: "
            f"{json.dumps(scene, ensure_ascii=False, separators=(',', ':'))}, produce a shot list for the scene. "
            "Return a JSON object with: scene_number (int), shots (list of 3 to 5 objects with shot_type, angle, "
            "lens, camera_movement, duration_sec). Respond only with the raw JSON object."
        )
        shot_list = call_hf_json(prompt)

        if not isinstance(shot_list, dict):
            raise ValueError("CinematographerAgent expected the Hugging Face model to return a JSON object.")

        shot_list["scene_number"] = scene_number
        shots = shot_list.get("shots")
        if not isinstance(shots, list):
            raise ValueError("CinematographerAgent expected the Hugging Face model to return a shots list.")

        shot_list["shots"] = shots
        return shot_list

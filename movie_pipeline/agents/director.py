from __future__ import annotations

from typing import Any

try:
    from movie_pipeline.agents.base import call_hf_json
except ImportError:  # pragma: no cover - direct execution fallback
    from agents.base import call_hf_json


class DirectorAgent:
    def run(self, movie_idea: str | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(movie_idea, dict):
            movie_idea = str(movie_idea.get("movie_idea", "")).strip()
        elif not isinstance(movie_idea, str):
            movie_idea = str(movie_idea)

        prompt = (
            f"You are a film director. Given this movie idea: '{movie_idea}', produce a JSON array of scenes. "
            "Each scene must have: scene_number (int), title (str), mood (str), pacing (str: slow/medium/fast), "
            "arc_position (str: setup/confrontation/resolution), setting (str). Respond only with the raw JSON array."
        )
        scenes = call_hf_json(prompt)

        if not isinstance(scenes, list):
            raise ValueError("DirectorAgent expected the Hugging Face model to return a JSON array of scenes.")

        return scenes

from __future__ import annotations

import json
from typing import Any

try:
    from movie_pipeline.agents.base import call_hf_json
except ImportError:  # pragma: no cover - direct execution fallback
    from agents.base import call_hf_json


class ScreenwriterAgent:
    def run(self, director_output: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(director_output, dict):
            director_output = director_output.get("director_output", [])
        if not isinstance(director_output, list):
            raise TypeError("ScreenwriterAgent expected a list of director scenes.")

        prompt = (
            "You are a screenwriter. Given these scenes: "
            f"{json.dumps(director_output, ensure_ascii=False, indent=2)}, write a script block for each. "
            "Output a JSON array where each item has: scene_number (int), setting (str), action_lines (list of str), "
            "dialogue (list of {character, line}). Respond only with the raw JSON array."
        )
        script_blocks = call_hf_json(prompt)

        if not isinstance(script_blocks, list):
            raise ValueError("ScreenwriterAgent expected the Hugging Face model to return a JSON array of script blocks.")

        return script_blocks

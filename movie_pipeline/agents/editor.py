from __future__ import annotations

import json
from typing import Any

try:
    from movie_pipeline.agents.base import call_hf_json
except ImportError:  # pragma: no cover - direct execution fallback
    from agents.base import call_hf_json


class EditorAgent:
    def run(
        self,
        director_output: list[dict[str, Any]] | dict[str, Any],
        screenwriter_output: list[dict[str, Any]] | None = None,
        cinematographer_output: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(director_output, dict):
            screenwriter_output = director_output.get("screenwriter_output", screenwriter_output or [])
            cinematographer_output = director_output.get("cinematographer_output", cinematographer_output or [])
            director_output = director_output.get("director_output", [])

        if not isinstance(director_output, list):
            raise TypeError("EditorAgent expected a list of director notes.")
        if not isinstance(screenwriter_output, list):
            raise TypeError("EditorAgent expected a list of script blocks.")
        if not isinstance(cinematographer_output, list):
            raise TypeError("EditorAgent expected a list of shot lists.")

        prompt = (
            "You are a film editor. Given director notes: "
            f"{json.dumps(director_output, ensure_ascii=False, indent=2)}, script: "
            f"{json.dumps(screenwriter_output, ensure_ascii=False, indent=2)}, and shot list: "
            f"{json.dumps(cinematographer_output, ensure_ascii=False, indent=2)}, create an edit plan for each scene. "
            "Output a JSON array where each item has: scene_number (int), cut_points (list of float seconds), "
            "transition_type (str), total_duration_sec (float). Respond only with the raw JSON array."
        )
        edit_plan = call_hf_json(prompt)

        if not isinstance(edit_plan, list):
            raise ValueError("EditorAgent expected the Hugging Face model to return a JSON array of edit plans.")

        return edit_plan

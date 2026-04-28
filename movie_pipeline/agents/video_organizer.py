from __future__ import annotations

import json
from typing import Any

try:
    from movie_pipeline.agents.base import call_hf_json
except ImportError:  # pragma: no cover - direct execution fallback
    from agents.base import call_hf_json


class VideoOrganizerAgent:
    def run(self, all_outputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if all_outputs is None:
            all_outputs = {}
        if not isinstance(all_outputs, dict):
            raise TypeError("VideoOrganizerAgent expected a dictionary of pipeline outputs.")

        prompt = (
            "You are a video production organizer. Given all pipeline outputs: "
            f"{json.dumps(all_outputs, ensure_ascii=False, indent=2)}, produce a final sequence manifest. "
            "Output a single JSON object with: sequence (list of {scene_number, title, video_prompt (str, written as a "
            "vivid visual description for a text-to-video model), order (int)}), final_runtime_sec (float), "
            "style_notes (str). Respond only with raw JSON."
        )
        manifest = call_hf_json(prompt)

        if not isinstance(manifest, dict):
            raise ValueError("VideoOrganizerAgent expected the Hugging Face model to return a JSON object.")

        return manifest

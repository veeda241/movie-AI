"""Deterministic offline planning when HF_TOKEN is not set."""

from __future__ import annotations

from typing import Any


def _idea_excerpt(movie_idea: str, max_len: int = 120) -> str:
    text = " ".join(movie_idea.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def local_director_scenes(movie_idea: str, scene_count: int = 3) -> list[dict[str, Any]]:
    idea = movie_idea.strip() or "Untitled story"
    templates = [
        {
            "title": "Opening",
            "mood": "curious",
            "pacing": "medium",
            "arc_position": "setup",
            "setting": f"INT/EXT — world of: {_idea_excerpt(idea, 80)}",
        },
        {
            "title": "Turning point",
            "mood": "tense",
            "pacing": "fast",
            "arc_position": "confrontation",
            "setting": f"EXT — conflict around: {_idea_excerpt(idea, 80)}",
        },
        {
            "title": "Resolution",
            "mood": "reflective",
            "pacing": "slow",
            "arc_position": "resolution",
            "setting": f"INT — aftermath of: {_idea_excerpt(idea, 80)}",
        },
    ]
    scenes: list[dict[str, Any]] = []
    for index in range(max(1, min(scene_count, len(templates)))):
        scene = dict(templates[index])
        scene["scene_number"] = index + 1
        scenes.append(scene)
    return scenes


def local_screenwriter_blocks(director_output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for scene in director_output:
        number = int(scene.get("scene_number", len(blocks) + 1))
        title = str(scene.get("title", f"Scene {number}"))
        setting = str(scene.get("setting", "INT. LOCATION - DAY"))
        mood = str(scene.get("mood", "dramatic"))
        blocks.append(
            {
                "scene_number": number,
                "setting": setting,
                "action_lines": [
                    f"{title}: the camera finds the subject in a {mood} atmosphere.",
                    "A decisive action advances the story.",
                    "The moment lands with a clear visual beat.",
                ],
                "dialogue": [
                    {"character": "PROTAGONIST", "line": f"This is where {title.lower()} begins."},
                ],
            }
        )
    return blocks


def local_cinematographer_shots(scene: dict[str, Any]) -> dict[str, Any]:
    number = int(scene.get("scene_number", 1))
    return {
        "scene_number": number,
        "shots": [
            {
                "shot_type": "wide",
                "angle": "eye-level",
                "lens": "35mm",
                "camera_movement": "slow push-in",
                "duration_sec": 2.0,
            },
            {
                "shot_type": "medium",
                "angle": "slight low",
                "lens": "50mm",
                "camera_movement": "tracking",
                "duration_sec": 2.0,
            },
            {
                "shot_type": "close-up",
                "angle": "eye-level",
                "lens": "85mm",
                "camera_movement": "static",
                "duration_sec": 1.5,
            },
        ],
    }


def local_editor_plans(
    director_output: list[dict[str, Any]],
    cinematographer_output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    shots_by_scene = {
        int(item.get("scene_number", 0)): item for item in cinematographer_output if isinstance(item, dict)
    }
    for scene in director_output:
        number = int(scene.get("scene_number", len(plans) + 1))
        shots = shots_by_scene.get(number, {}).get("shots", [])
        total = 0.0
        cut_points: list[float] = []
        for shot in shots:
            if isinstance(shot, dict):
                total += float(shot.get("duration_sec", 1.5))
                cut_points.append(round(total, 2))
        if not cut_points:
            total = 4.0
            cut_points = [1.5, 3.0, 4.0]
        plans.append(
            {
                "scene_number": number,
                "cut_points": cut_points,
                "transition_type": "cut",
                "total_duration_sec": float(total),
            }
        )
    return plans


def local_organizer_manifest(all_outputs: dict[str, Any]) -> dict[str, Any]:
    movie_idea = str(all_outputs.get("movie_idea", "cinematic scene"))
    director_output = all_outputs.get("director_output", [])
    if not isinstance(director_output, list):
        director_output = []

    sequence: list[dict[str, Any]] = []
    runtime = 0.0
    editor_output = all_outputs.get("editor_output", [])
    duration_by_scene: dict[int, float] = {}
    if isinstance(editor_output, list):
        for item in editor_output:
            if isinstance(item, dict):
                duration_by_scene[int(item.get("scene_number", 0))] = float(
                    item.get("total_duration_sec", 4.0)
                )

    for index, scene in enumerate(director_output):
        if not isinstance(scene, dict):
            continue
        number = int(scene.get("scene_number", index + 1))
        title = str(scene.get("title", f"Scene {number}"))
        mood = str(scene.get("mood", "cinematic"))
        setting = str(scene.get("setting", "dramatic location"))
        video_prompt = (
            f"Cinematic {mood} shot, {setting}. Story beat: {title}. "
            f"Based on: {_idea_excerpt(movie_idea, 100)}. Photoreal film look, natural motion."
        )
        sequence.append(
            {
                "scene_number": number,
                "title": title,
                "video_prompt": video_prompt,
                "order": index + 1,
            }
        )
        runtime += duration_by_scene.get(number, 4.0)

    return {
        "sequence": sequence,
        "final_runtime_sec": float(runtime or max(4.0, len(sequence) * 4.0)),
        "style_notes": "Local offline planning (set HF_TOKEN for LLM agents).",
    }


def has_hf_token() -> bool:
    import os

    return bool(os.environ.get("HF_TOKEN", "").strip())

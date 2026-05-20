from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from movie_pipeline.pipeline.scene_packet import ScenePacket


def build_image_prompt(packet: ScenePacket) -> str:
    shot_language = _summarize_shots(packet.shots)
    return (
        f"{packet.title}. {packet.video_prompt}. Setting: {packet.setting}. "
        f"Mood: {packet.mood}. {shot_language} "
        "Cinematic production still, detailed environment, film lighting."
    ).strip()


def build_video_prompt(packet: ScenePacket) -> str:
    shot_language = _summarize_shots(packet.shots)
    edit_language = _summarize_edit_plan(packet.edit_plan)
    return (
        f"{packet.video_prompt}. Setting: {packet.setting}. Mood: {packet.mood}. "
        f"{shot_language} {edit_language} Smooth temporal motion, coherent characters, cinematic camera."
    ).strip()


def _summarize_shots(shots: list[Any]) -> str:
    fragments: list[str] = []
    for shot in shots[:4]:
        if not isinstance(shot, dict):
            continue
        shot_type = str(shot.get("shot_type", "")).strip()
        angle = str(shot.get("angle", "")).strip()
        lens = str(shot.get("lens", "")).strip()
        movement = str(shot.get("camera_movement", "")).strip()
        fragment = ", ".join(part for part in [shot_type, angle, lens, movement] if part)
        if fragment:
            fragments.append(fragment)

    if not fragments:
        return ""
    return "Camera plan: " + "; ".join(fragments) + "."


def _summarize_edit_plan(edit_plan: dict[str, Any]) -> str:
    if not edit_plan:
        return ""
    pacing = str(edit_plan.get("pacing", "")).strip()
    transition = str(edit_plan.get("transition", "")).strip()
    parts = []
    if pacing:
        parts.append(f"{pacing} pacing")
    if transition:
        parts.append(f"{transition} transitions")
    if not parts:
        return ""
    return "Edit rhythm: " + ", ".join(parts) + "."

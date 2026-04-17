from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ScenePacket:
    scene_number: int
    title: str
    mood: str
    setting: str
    script: dict[str, Any]
    shots: list[Any]
    edit_plan: dict[str, Any]
    video_prompt: str
    video_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

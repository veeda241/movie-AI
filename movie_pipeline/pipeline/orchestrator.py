from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from movie_pipeline.agents.cinematographer import CinematographerAgent
from movie_pipeline.agents.director import DirectorAgent
from movie_pipeline.agents.editor import EditorAgent
from movie_pipeline.agents.screenwriter import ScreenwriterAgent
from movie_pipeline.agents.video_organizer import VideoOrganizerAgent
from movie_pipeline.pipeline.scene_packet import ScenePacket
from movie_pipeline.video.motif_client import MotifClient


class Orchestrator:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.director = DirectorAgent()
        self.screenwriter = ScreenwriterAgent()
        self.cinematographer = CinematographerAgent()
        self.editor = EditorAgent()
        self.video_organizer = VideoOrganizerAgent()
        self.video_client = MotifClient()
        self.last_organizer_output: dict[str, Any] = {}

    def run(
        self,
        movie_idea: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[ScenePacket]:
        context: dict[str, Any] = {"movie_idea": movie_idea}

        self._emit_progress(progress_callback, "[1/6] Director agent: generating scene outline")
        director_output = self.director.run(movie_idea)
        context["director_output"] = director_output
        self._emit_progress(progress_callback, f"[1/6] Director agent done -> {len(director_output)} scenes")

        self._emit_progress(progress_callback, "[2/6] Screenwriter agent: drafting scene scripts")
        screenwriter_output = self.screenwriter.run(director_output)
        context["screenwriter_output"] = screenwriter_output
        self._emit_progress(progress_callback, "[2/6] Screenwriter agent done")

        self._emit_progress(progress_callback, "[3/6] Cinematographer agent: building shot plans")
        cinematographer_output = self.cinematographer.run(screenwriter_output)
        context["cinematographer_output"] = cinematographer_output
        self._emit_progress(progress_callback, "[3/6] Cinematographer agent done")

        self._emit_progress(progress_callback, "[4/6] Editor agent: assembling edit plans")
        editor_output = self.editor.run(director_output, screenwriter_output, cinematographer_output)
        context["editor_output"] = editor_output
        self._emit_progress(progress_callback, "[4/6] Editor agent done")

        self._emit_progress(progress_callback, "[5/6] Video organizer: producing sequence manifest")
        organizer_output = self.video_organizer.run(context)
        context["organizer_output"] = organizer_output
        self.last_organizer_output = organizer_output
        self._emit_progress(progress_callback, "[5/6] Video organizer done")

        self._emit_progress(progress_callback, "[5/6] Building scene packets")
        packets = self._build_scene_packets(context, organizer_output)
        self._emit_progress(progress_callback, f"[5/6] Built {len(packets)} scene packets")
        for packet in packets:
            self._write_scene_packet(packet)

        self._emit_progress(progress_callback, f"[6/6] Generating videos for {len(packets)} scenes")
        self._generate_videos(packets, progress_callback)

        if packets:
            packets[-1].edit_plan["total_duration_sec"] = float(
                organizer_output.get(
                    "final_runtime_sec",
                    packets[-1].edit_plan.get("total_duration_sec", 0.0),
                )
            )
            self._write_scene_packet(packets[-1])

        self._emit_progress(progress_callback, "[Done] Pipeline complete")

        return packets

    def _emit_progress(self, progress_callback: Callable[[str], None] | None, message: str) -> None:
        print(message, flush=True)
        if progress_callback is not None:
            progress_callback(message)

    def _build_scene_packets(
        self,
        context: dict[str, Any],
        organizer_output: dict[str, Any],
    ) -> list[ScenePacket]:
        director_output = context.get("director_output", [])
        screenwriter_output = context.get("screenwriter_output", [])
        cinematographer_output = context.get("cinematographer_output", [])
        editor_output = context.get("editor_output", [])

        if not isinstance(director_output, list):
            raise TypeError("director_output must be a list.")
        if not isinstance(screenwriter_output, list):
            raise TypeError("screenwriter_output must be a list.")
        if not isinstance(cinematographer_output, list):
            raise TypeError("cinematographer_output must be a list.")
        if not isinstance(editor_output, list):
            raise TypeError("editor_output must be a list.")

        sequence = organizer_output.get("sequence", [])
        if not isinstance(sequence, list):
            raise TypeError("VideoOrganizerAgent output must include a sequence list.")

        director_by_scene = self._index_by_scene_number(director_output, "director_output")
        script_by_scene = self._index_by_scene_number(screenwriter_output, "screenwriter_output")
        shots_by_scene = self._index_by_scene_number(cinematographer_output, "cinematographer_output")
        edit_plan_by_scene = self._index_by_scene_number(editor_output, "editor_output")

        packets: list[ScenePacket] = []
        ordered_sequence = sorted(
            sequence,
            key=lambda item: int(item.get("order", item.get("scene_number", 0))),
        )

        for sequence_item in ordered_sequence:
            if not isinstance(sequence_item, dict):
                raise TypeError("Each sequence item must be a dictionary.")

            scene_number = int(sequence_item.get("scene_number", 0))
            director_scene = director_by_scene.get(scene_number)
            script_scene = script_by_scene.get(scene_number)
            shots_scene = shots_by_scene.get(scene_number)
            edit_scene = edit_plan_by_scene.get(scene_number)

            if director_scene is None:
                raise ValueError(f"Missing director output for scene {scene_number}.")
            if script_scene is None:
                raise ValueError(f"Missing screenwriter output for scene {scene_number}.")
            if shots_scene is None:
                raise ValueError(f"Missing cinematographer output for scene {scene_number}.")
            if edit_scene is None:
                raise ValueError(f"Missing editor output for scene {scene_number}.")

            packets.append(
                ScenePacket(
                    scene_number=scene_number,
                    title=str(sequence_item.get("title", director_scene.get("title", ""))),
                    mood=str(director_scene.get("mood", "")),
                    setting=str(director_scene.get("setting", script_scene.get("setting", ""))),
                    script=dict(script_scene),
                    shots=list(shots_scene.get("shots", [])),
                    edit_plan=dict(edit_scene),
                    video_prompt=str(sequence_item.get("video_prompt", "")),
                )
            )

        return packets

    def _index_by_scene_number(self, items: list[dict[str, Any]], source_name: str) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                raise TypeError(f"Each item in {source_name} must be a dictionary.")
            if "scene_number" not in item:
                raise ValueError(f"Each item in {source_name} must include scene_number.")
            index[int(item["scene_number"])] = item
        return index

    def _write_scene_packet(self, packet: ScenePacket) -> None:
        packet_path = self.output_dir / f"scene_{packet.scene_number}_packet.json"
        packet_path.write_text(
            json.dumps(asdict(packet), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _generate_videos(
        self,
        packets: list[ScenePacket],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not packets:
            return

        for packet in packets:
            self._generate_video_for_packet(packet, progress_callback)

    def _generate_video_for_packet(
        self,
        packet: ScenePacket,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._emit_progress(progress_callback, f"[Video] Scene {packet.scene_number}: generating video")
        packet.video_path = self.video_client.generate(packet.video_prompt, packet.scene_number)
        self._write_scene_packet(packet)
        if packet.video_path:
            self._emit_progress(progress_callback, f"[Video] Scene {packet.scene_number}: saved {packet.video_path}")
        else:
            self._emit_progress(progress_callback, f"[Video] Scene {packet.scene_number}: no video produced")

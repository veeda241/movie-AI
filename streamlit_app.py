from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import streamlit as st

from movie_pipeline.pipeline.orchestrator import Orchestrator
from movie_pipeline.pipeline.scene_packet import ScenePacket

APP_TITLE = "Movie AI Studio"
APP_SUBTITLE = "Plan scenes with Hugging Face agents and render video stubs with Motif-Video-2B."


def _ensure_state() -> None:
    if "scene_packets" not in st.session_state:
        st.session_state.scene_packets = []
    if "organizer_output" not in st.session_state:
        st.session_state.organizer_output = {}
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []
    if "movie_idea" not in st.session_state:
        st.session_state.movie_idea = ""
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""


def _missing_env_vars() -> list[str]:
    required_vars = ["HF_TOKEN"]
    return [name for name in required_vars if not os.environ.get(name)]


def _scene_packets_to_dicts(scene_packets: list[ScenePacket]) -> list[dict[str, Any]]:
    return [packet.to_dict() for packet in scene_packets]


def _read_json_file(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.warning(f"Failed to parse JSON from packet file '{file_path}': {exc}")
        return None


def _render_summary(scene_packets: list[ScenePacket], organizer_output: dict[str, Any]) -> None:
    runtime = organizer_output.get("final_runtime_sec")
    style_notes = organizer_output.get("style_notes", "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Scenes", len(scene_packets))
    col2.metric("Runtime", f"{runtime}s" if runtime is not None else "Unknown")
    col3.metric("Videos", sum(1 for packet in scene_packets if packet.video_path))

    if style_notes:
        st.info(style_notes)


def _render_scene_packet(packet: ScenePacket) -> None:
    packet_dict = packet.to_dict()
    with st.expander(f"Scene {packet.scene_number}: {packet.title}", expanded=True):
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Scene Details")
            st.write(f"Mood: {packet.mood}")
            st.write(f"Setting: {packet.setting}")
            st.write(f"Video prompt: {packet.video_prompt}")

            if packet.script:
                st.markdown("**Script**")
                st.json(packet.script)
            if packet.shots:
                st.markdown("**Shots**")
                st.json(packet.shots)
            if packet.edit_plan:
                st.markdown("**Edit plan**")
                st.json(packet.edit_plan)

        with right:
            st.subheader("Media")
            if packet.video_path and Path(packet.video_path).exists():
                st.video(packet.video_path)
                st.caption(packet.video_path)
            elif packet.video_path:
                st.warning(f"Video path reported, but file was not found: {packet.video_path}")
            else:
                st.info("No video generated for this scene yet.")

            st.markdown("**Raw packet**")
            st.json(packet_dict)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_state()

    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)

    with st.sidebar:
        st.header("Pipeline")
        st.write("Enter a movie idea, run the Hugging Face agent chain, and inspect each generated scene packet.")
        st.write("Environment variables required:")
        st.code("HF_TOKEN")
        missing_env_vars = _missing_env_vars()
        if missing_env_vars:
            st.warning(f"Missing environment variables: {', '.join(missing_env_vars)}")
        if st.button("Clear results", use_container_width=True):
            st.session_state.scene_packets = []
            st.session_state.organizer_output = {}
            st.session_state.processing_log = []
            st.session_state.last_error = ""
            st.session_state.movie_idea = ""
            st.rerun()

    st.session_state.movie_idea = st.text_area(
        "Movie idea",
        value=st.session_state.movie_idea,
        height=140,
        placeholder="A disgraced astronaut returns to Earth to uncover why the moon is broadcasting her childhood memories.",
    )

    run_clicked = st.button("Generate movie", type="primary", use_container_width=True)

    processing_panel = st.container()
    with processing_panel:
        st.subheader("Agent Processing")
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        if st.session_state.processing_log:
            status_placeholder.info(st.session_state.processing_log[-1])
            log_placeholder.code("\n".join(st.session_state.processing_log), language="text")
        else:
            status_placeholder.caption("The pipeline status will appear here while the agents run.")

    if run_clicked:
        idea = st.session_state.movie_idea.strip()
        if not idea:
            st.session_state.last_error = "Enter a movie idea before running the pipeline."
        else:
            st.session_state.last_error = ""
            st.session_state.processing_log = []
            progress_log = st.session_state.processing_log

            def update_progress(message: str) -> None:
                progress_log.append(message)
                status_placeholder.info(message)
                log_placeholder.code("\n".join(progress_log), language="text")

            update_progress("Starting pipeline...")
            with st.spinner("Running agents and generating videos..."):
                try:
                    orchestrator = Orchestrator()
                    scene_packets = orchestrator.run(idea, progress_callback=update_progress)
                    st.session_state.scene_packets = scene_packets
                    st.session_state.organizer_output = orchestrator.last_organizer_output
                    update_progress("Pipeline finished successfully.")
                except Exception as exc:
                    st.session_state.scene_packets = []
                    st.session_state.organizer_output = {}
                    st.session_state.last_error = str(exc)
                    update_progress(f"Pipeline failed: {exc}")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    scene_packets = st.session_state.scene_packets
    organizer_output = st.session_state.organizer_output

    if scene_packets:
        _render_summary(scene_packets, organizer_output)

        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.subheader("Sequence manifest")
            st.json(organizer_output or {"message": "No organizer output available yet."})
        with top_right:
            st.subheader("Scene packet index")
            st.write([packet.scene_number for packet in scene_packets])

        for packet in scene_packets:
            _render_scene_packet(packet)

        with st.expander("All scene packets as JSON", expanded=False):
            st.json(_scene_packets_to_dicts(scene_packets))

        with st.expander("Loaded packet files from output", expanded=False):
            for packet in scene_packets:
                file_path = Path("movie_pipeline") / "output" / f"scene_{packet.scene_number}_packet.json"
                loaded_packet = _read_json_file(str(file_path))
                if loaded_packet is not None:
                    st.markdown(f"**{file_path.as_posix()}**")
                    st.json(loaded_packet)
    else:
        st.info("Run the pipeline to generate a scene breakdown and video previews.")


if __name__ == "__main__":
    main()

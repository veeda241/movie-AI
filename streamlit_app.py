from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from movie_pipeline.pipeline.orchestrator import Orchestrator
from movie_pipeline.pipeline.scene_packet import ScenePacket

APP_TITLE = "Movie AI Studio"
APP_SUBTITLE = (
    "Plan scenes with Hugging Face agents, render cinematic video fallbacks, "
    "and review the whole production in one dashboard."
)
FEATURE_TAGS = (
    "Director AI",
    "Screenwriter AI",
    "Cinematographer AI",
    "Editor AI",
    "Video Organizer",
)

THEME_CSS = """
<style>
    :root {
        --bg: #060814;
        --panel: rgba(12, 18, 34, 0.82);
        --panel-strong: rgba(18, 26, 48, 0.92);
        --border: rgba(255, 255, 255, 0.08);
        --text: #edf2ff;
        --muted: #aab4d6;
        --accent: #7f8cff;
        --accent-2: #d66eff;
        --accent-3: #ffc857;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(127, 140, 255, 0.22), transparent 32%),
            radial-gradient(circle at top right, rgba(214, 110, 255, 0.18), transparent 28%),
            linear-gradient(180deg, #070814 0%, #090d1d 38%, #05060b 100%);
        color: var(--text);
    }

    .movie-hero {
        padding: 2rem 2rem 1.5rem;
        border: 1px solid var(--border);
        border-radius: 1.5rem;
        background:
            linear-gradient(135deg, rgba(20, 28, 53, 0.96), rgba(7, 10, 21, 0.92)),
            radial-gradient(circle at top right, rgba(255, 200, 87, 0.14), transparent 18%);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        margin-bottom: 1rem;
    }

    .movie-hero h1 {
        color: var(--text);
        font-size: clamp(2.2rem, 5vw, 4.4rem);
        line-height: 0.95;
        margin: 0.25rem 0 0.65rem;
        letter-spacing: -0.05em;
    }

    .movie-hero p {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        max-width: 62rem;
        margin-bottom: 0;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--accent-3);
        margin-bottom: 0.35rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .chip {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 0.42rem 0.85rem;
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 0.82rem;
    }

    .movie-panel {
        border: 1px solid var(--border);
        border-radius: 1.25rem;
        padding: 1.1rem 1.2rem;
        background: var(--panel);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
        margin-bottom: 1rem;
    }

    .movie-panel h3,
    .movie-panel h4 {
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .scene-card {
        border: 1px solid var(--border);
        border-radius: 1.25rem;
        padding: 1rem 1.1rem;
        background: var(--panel-strong);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .scene-kicker {
        font-size: 0.73rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--accent-3);
        margin-bottom: 0.4rem;
    }

    .scene-title {
        color: var(--text);
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .scene-meta {
        color: var(--muted);
        margin-bottom: 0;
    }

    .movie-empty {
        text-align: center;
        padding: 2rem 1.5rem;
    }

    .movie-empty h3 {
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .movie-empty p {
        color: var(--muted);
        margin-bottom: 0;
    }

    .movie-caption {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .movie-log {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.85rem;
    }
</style>
"""


def _ensure_state() -> None:
    defaults = {
        "scene_packets": [],
        "organizer_output": {},
        "processing_log": [],
        "movie_idea": "",
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _missing_env_vars() -> list[str]:
    required_vars = ["HF_TOKEN"]
    return [name for name in required_vars if not os.environ.get(name)]


def _format_runtime(seconds: Any) -> str:
    if seconds is None:
        return "Unknown"

    try:
        total_seconds = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return "Unknown"

    minutes, remainder = divmod(int(round(total_seconds)), 60)
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


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


def _manifest_rows(scene_packets: list[ScenePacket], organizer_output: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = organizer_output.get("sequence", [])
    if isinstance(sequence, list) and sequence:
        rows: list[dict[str, Any]] = []
        for item in sequence:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "Order": item.get("order", ""),
                    "Scene": item.get("scene_number", ""),
                    "Title": item.get("title", ""),
                    "Prompt": item.get("video_prompt", ""),
                }
            )
        return rows

    return [
        {
            "Order": index + 1,
            "Scene": packet.scene_number,
            "Title": packet.title,
            "Prompt": packet.video_prompt,
        }
        for index, packet in enumerate(scene_packets)
    ]


def _build_summary(scene_packets: list[ScenePacket], organizer_output: dict[str, Any]) -> dict[str, Any]:
    scene_count = len(organizer_output.get("sequence", [])) if isinstance(organizer_output.get("sequence"), list) else len(scene_packets)
    runtime = organizer_output.get("final_runtime_sec")
    if runtime is None and scene_packets:
        runtime = sum(float(packet.edit_plan.get("total_duration_sec", 0.0)) for packet in scene_packets)

    return {
        "scene_count": scene_count or len(scene_packets),
        "shot_count": sum(len(packet.shots) for packet in scene_packets),
        "video_count": sum(1 for packet in scene_packets if packet.video_path),
        "runtime_label": _format_runtime(runtime),
        "style_notes": str(organizer_output.get("style_notes", "")).strip(),
    }


def _apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def _render_hero() -> None:
    chips = "".join(f"<span class='chip'>{tag}</span>" for tag in FEATURE_TAGS)
    st.markdown(
        f"""
        <div class="movie-hero">
            <div class="eyebrow">CINEMATIC AI PRODUCTION STUDIO</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <div class="chip-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(orchestrator: Orchestrator) -> None:
    with st.sidebar:
        st.markdown("### Production Console")
        st.caption("Run the agent chain, inspect saved packets, and manage generated assets.")
        st.caption(f"Output directory: `{orchestrator.output_dir}`")

        missing_env_vars = _missing_env_vars()
        if missing_env_vars:
            st.warning(f"Missing environment variables: {', '.join(missing_env_vars)}")
        else:
            st.success("Backend token ready.")

        if st.button("Load saved project", use_container_width=True):
            saved_project = orchestrator.load_saved_project()
            st.session_state.scene_packets = saved_project["scene_packets"]
            st.session_state.organizer_output = saved_project["organizer_output"]
            if st.session_state.scene_packets:
                st.session_state.processing_log = [
                    f"Loaded {len(st.session_state.scene_packets)} saved scene packets from disk."
                ]
            else:
                st.session_state.processing_log = ["No saved project found in movie_pipeline/output."]
            st.session_state.last_error = ""

        if st.button("Clear workspace", use_container_width=True):
            st.session_state.scene_packets = []
            st.session_state.organizer_output = {}
            st.session_state.processing_log = []
            st.session_state.last_error = ""
            st.session_state.movie_idea = ""
            st.rerun()


def _render_entry_controls() -> bool:
    st.markdown(
        """
        <div class="movie-panel">
            <h3>Generate a new cut</h3>
            <p class="movie-caption">Write a logline, then let the director, writer, cinematographer, editor, and video organizer build the sequence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.movie_idea = st.text_area(
        "Movie idea",
        value=st.session_state.movie_idea,
        height=140,
        placeholder="A disgraced astronaut returns to Earth to uncover why the moon is broadcasting her childhood memories.",
    )

    return st.button("Generate cinematic cut", type="primary", use_container_width=True)


def _render_console() -> tuple[Any, Any]:
    st.markdown(
        """
        <div class="movie-panel">
            <h3>Live Production Console</h3>
            <p class="movie-caption">Pipeline updates appear here while the backend runs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()
    log_placeholder = st.empty()
    if st.session_state.processing_log:
        status_placeholder.info(st.session_state.processing_log[-1])
        log_placeholder.code("\n".join(st.session_state.processing_log), language="text")
    else:
        status_placeholder.caption("The pipeline status will appear here.")

    return status_placeholder, log_placeholder


def _render_summary(scene_packets: list[ScenePacket], organizer_output: dict[str, Any]) -> None:
    summary = _build_summary(scene_packets, organizer_output)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scenes", summary["scene_count"])
    col2.metric("Shots", summary["shot_count"])
    col3.metric("Videos", summary["video_count"])
    col4.metric("Runtime", summary["runtime_label"])

    if summary["style_notes"]:
        st.info(summary["style_notes"])


def _render_manifest(scene_packets: list[ScenePacket], organizer_output: dict[str, Any]) -> None:
    st.markdown("### Sequence manifest")
    st.table(_manifest_rows(scene_packets, organizer_output))


def _render_scene_packet(packet: ScenePacket) -> None:
    packet_dict = packet.to_dict()
    st.markdown(
        f"""
        <div class="scene-card">
            <div class="scene-kicker">Scene {packet.scene_number}</div>
            <div class="scene-title">{packet.title}</div>
            <p class="scene-meta">{packet.mood} · {packet.setting}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("**Video prompt**")
        st.code(packet.video_prompt or "No video prompt generated.", language="text")
        if packet.image_prompt:
            st.markdown("**Image prompt**")
            st.code(packet.image_prompt, language="text")
        st.markdown("**Script**")
        st.json(packet.script)
        st.markdown("**Shots**")
        st.json(packet.shots)
        st.markdown("**Edit plan**")
        st.json(packet.edit_plan)

    with right:
        st.markdown("**Media**")
        if packet.image_path and Path(packet.image_path).exists():
            st.image(packet.image_path, caption=Path(packet.image_path).name, use_container_width=True)
        elif packet.image_path:
            st.warning(f"Image path reported, but file was not found: {packet.image_path}")

        if packet.video_path and Path(packet.video_path).exists():
            st.video(packet.video_path)
            st.caption(Path(packet.video_path).name)
        elif packet.video_path:
            st.warning(f"Video path reported, but file was not found: {packet.video_path}")
        else:
            st.info("No video generated for this scene yet.")

        st.download_button(
            label=f"Download scene {packet.scene_number} packet",
            data=json.dumps(packet_dict, indent=2, ensure_ascii=False),
            file_name=f"scene_{packet.scene_number}_packet.json",
            mime="application/json",
            use_container_width=True,
            key=f"download-scene-{packet.scene_number}",
        )

        with st.expander("Raw packet JSON", expanded=False):
            st.json(packet_dict)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
    _ensure_state()
    _apply_theme()

    try:
        orchestrator = Orchestrator()
    except Exception as exc:
        st.error(f"Failed to initialize backend orchestrator: {exc}")
        st.stop()

    _render_sidebar(orchestrator)
    _render_hero()

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        generate_clicked = _render_entry_controls()
    with right:
        st.markdown(
            """
            <div class="movie-panel">
                <h3>Backend status</h3>
                <p class="movie-caption">The orchestrator writes packets and manifests to disk so the dashboard can restore a saved cut.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("A saved manifest and scene packet JSON files are kept in `movie_pipeline/output`.")

    status_placeholder, log_placeholder = _render_console()

    if generate_clicked:
        idea = st.session_state.movie_idea.strip()
        if not idea:
            st.session_state.last_error = "Enter a movie idea before running the pipeline."
            status_placeholder.error(st.session_state.last_error)
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

        top_left, top_right = st.columns([1.1, 0.9], gap="large")
        with top_left:
            _render_manifest(scene_packets, organizer_output)
        with top_right:
            st.markdown("### Project files")
            st.download_button(
                label="Download manifest JSON",
                data=json.dumps(organizer_output or {"message": "No organizer output available."}, indent=2, ensure_ascii=False),
                file_name="sequence_manifest.json",
                mime="application/json",
                use_container_width=True,
                key="download-manifest",
            )
            st.download_button(
                label="Download all packets JSON",
                data=json.dumps(_scene_packets_to_dicts(scene_packets), indent=2, ensure_ascii=False),
                file_name="scene_packets.json",
                mime="application/json",
                use_container_width=True,
                key="download-packets",
            )
            st.caption(f"Saved packet path pattern: `movie_pipeline/output/scene_<n>_packet.json`")

        for packet in scene_packets:
            _render_scene_packet(packet)

        with st.expander("Loaded packet files from output", expanded=False):
            for packet in scene_packets:
                file_path = Path("movie_pipeline") / "output" / f"scene_{packet.scene_number}_packet.json"
                loaded_packet = _read_json_file(str(file_path))
                if loaded_packet is not None:
                    st.markdown(f"**{file_path.as_posix()}**")
                    st.json(loaded_packet)
    else:
        st.markdown(
            """
            <div class="movie-panel movie-empty">
                <h3>Ready to cut the first scene</h3>
                <p>Enter a movie idea, generate the project, and the dashboard will fill with scene cards, manifests, and playable media.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

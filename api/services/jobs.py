from __future__ import annotations

import json
import sys
from pathlib import Path

# The patched Wan2.1-T2V-1.3B generator lives only in the top-level video_lab/
# package, not the older copy bundled inside this directory (movie-AI-lab/).
# Walk up from this file to find the directory containing video_lab/__init__.py
# and prefer it on sys.path so `import video_lab` resolves to the working Wan
# pipeline regardless of which copy of the api package is launched.
_file_dir = Path(__file__).resolve().parent
for _candidate in (_file_dir.parents[0], *_file_dir.parents):
    if (_candidate / "video_lab" / "__init__.py").exists() and not _candidate.name.endswith("movie-AI-lab"):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from sqlalchemy.orm import Session

from api.config import settings
from api.models import Asset, AssetKind, Job, JobKind, JobStatus, Project, User
from api.services import credits as credit_service
from movie_pipeline.models.image_client import ImageGenerationClient
from movie_pipeline.pipeline.orchestrator import Orchestrator
from movie_pipeline.video.motif_client import MotifClient

# Model id surfaced in the web UI for the local Wan 2.1 1.3B generator.
WAN_1_3B_MODEL_ID = "wan-2.1-1.3b"


def project_storage(user_id: str, project_id: str) -> Path:
    path = settings.storage_path / user_id / project_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(exist_ok=True)
    (path / "videos").mkdir(exist_ok=True)
    (path / "scenes").mkdir(exist_ok=True)
    return path


def append_job_event(db: Session, job: Job, message: str) -> None:
    events = json.loads(job.events_json or "[]")
    events.append(message)
    job.events_json = json.dumps(events)
    db.add(job)
    db.commit()
    db.refresh(job)


def set_job_status(db: Session, job: Job, status: JobStatus, error: str = "") -> Job:
    job.status = status
    if error:
        job.error = error
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def create_asset(
    db: Session,
    *,
    project: Project,
    user: User,
    kind: AssetKind,
    prompt: str,
    file_path: str,
    mime_type: str,
    meta: dict | None = None,
) -> Asset:
    asset = Asset(
        project_id=project.id,
        owner_id=user.id,
        kind=kind,
        prompt=prompt,
        file_path=file_path,
        mime_type=mime_type,
        meta_json=json.dumps(meta or {}),
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def estimate_movie_cost(scene_hint: int = 3) -> int:
    return max(1, scene_hint) * settings.credit_cost_movie_scene


def create_job(
    db: Session,
    *,
    user: User,
    project: Project,
    kind: JobKind,
    prompt: str,
    model: str,
    credits_charged: int,
    payload: dict | None = None,
) -> Job:
    credit_service.charge_credits(
        db,
        user,
        credits_charged,
        reason=f"job:{kind.value}",
    )
    job = Job(
        project_id=project.id,
        owner_id=user.id,
        kind=kind,
        status=JobStatus.queued,
        prompt=prompt,
        model=model,
        credits_charged=credits_charged,
        events_json=json.dumps(["Job queued"]),
        payload_json=json.dumps(payload or {}),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def run_image_job(db: Session, job_id: str) -> None:
    job = db.get(Job, job_id)
    if job is None:
        return
    user = db.get(User, job.owner_id)
    project = db.get(Project, job.project_id)
    if user is None or project is None:
        set_job_status(db, job, JobStatus.failed, "Missing user or project")
        return

    set_job_status(db, job, JobStatus.running)
    append_job_event(db, job, "Generating image…")
    try:
        out_dir = project_storage(user.id, project.id)
        out_path = out_dir / "images" / f"{job.id}.png"
        client = ImageGenerationClient(output_dir=out_dir / "images")
        path = client.generate_still(job.prompt, output_path=out_path)
        asset = create_asset(
            db,
            project=project,
            user=user,
            kind=AssetKind.image,
            prompt=job.prompt,
            file_path=path,
            mime_type="image/png",
            meta={"model": job.model, "job_id": job.id},
        )
        job.result_asset_ids = json.dumps([asset.id])
        append_job_event(db, job, f"Image saved: {asset.id}")
        set_job_status(db, job, JobStatus.succeeded)
    except Exception as exc:
        append_job_event(db, job, f"Failed: {exc}")
        set_job_status(db, job, JobStatus.failed, str(exc))


def _generate_with_local_wan(db: Session, job: Job, out_path: Path) -> str:
    """Run the patched local Wan2.1-T2V-1.3B pipeline (`generate_finetune_video`)
    and write the resulting mp4 to `out_path` (returns the Wan samples path; the
    caller then copies it into project storage)."""
    import shutil

    # Lazy import keeps the API light and surfaces clean errors if the user
    # picks Wan without CUDA / deps installed.
    from video_lab.infer.finetune_generate import generate_finetune_video

    def _log(msg: str) -> None:
        # Best-effort DB logger; events land in the same job row so the front-end
        # progress drawer can show "Loading Wan: ...", "Loaded PEFT LoRA ...", etc.
        try:
            append_job_event(db, job, msg)
        except Exception:
            pass

    seed = int.from_bytes(job.id.encode("utf-8")[:8], "big") % (2**32)

    wan_path = generate_finetune_video(
        job.prompt,
        seed=seed,
        steps=30,
        frames=81,
        fps=16,
        height=480,
        width=832,
        use_lora=False,
        log_fn=_log,
    )

    if not wan_path or not Path(wan_path).exists():
        raise RuntimeError(f"Wan generator returned no file: {wan_path!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wan_path, out_path)
    return str(out_path)


def run_video_job(db: Session, job_id: str) -> None:
    job = db.get(Job, job_id)
    if job is None:
        return
    user = db.get(User, job.owner_id)
    project = db.get(Project, job.project_id)
    if user is None or project is None:
        set_job_status(db, job, JobStatus.failed, "Missing user or project")
        return

    set_job_status(db, job, JobStatus.running)
    append_job_event(db, job, "Generating video clip...")
    try:
        out_dir = project_storage(user.id, project.id)
        existing_clips = (
            db.query(Asset)
            .filter(
                Asset.project_id == project.id,
                Asset.owner_id == user.id,
                Asset.kind == AssetKind.video,
            )
            .all()
        )
        clip_index = 1
        for asset in existing_clips:
            meta = json.loads(asset.meta_json or "{}")
            if meta.get("role") == "film":
                continue
            clip_index += 1

        out_path = out_dir / "videos" / f"clip_{clip_index:02d}_{job.id[:8]}.mp4"
        if job.model == WAN_1_3B_MODEL_ID:
            path = _generate_with_local_wan(db, job, out_path)
        else:
            client = MotifClient(output_dir=out_dir / "videos")
            path = client.generate(job.prompt, scene_number=clip_index, output_path=out_path)
        if not path:
            raise RuntimeError("Video generation produced no file")
        asset = create_asset(
            db,
            project=project,
            user=user,
            kind=AssetKind.video,
            prompt=job.prompt,
            file_path=path,
            mime_type="video/mp4",
            meta={
                "model": job.model,
                "job_id": job.id,
                "role": "clip",
                "clip_index": clip_index,
            },
        )
        job.result_asset_ids = json.dumps([asset.id])
        append_job_event(db, job, f"Clip {clip_index} saved: {asset.id}")
        set_job_status(db, job, JobStatus.succeeded)
    except Exception as exc:
        append_job_event(db, job, f"Failed: {exc}")
        set_job_status(db, job, JobStatus.failed, str(exc))


def run_assemble_job(db: Session, job_id: str) -> None:
    """Stitch ordered video clips into one film with ffmpeg concat."""
    import subprocess
    import tempfile

    import imageio_ffmpeg

    job = db.get(Job, job_id)
    if job is None:
        return
    user = db.get(User, job.owner_id)
    project = db.get(Project, job.project_id)
    if user is None or project is None:
        set_job_status(db, job, JobStatus.failed, "Missing user or project")
        return

    set_job_status(db, job, JobStatus.running)
    payload = json.loads(job.payload_json or "{}")
    asset_ids = payload.get("asset_ids") or []
    if len(asset_ids) < 2:
        set_job_status(db, job, JobStatus.failed, "Need at least 2 clips to assemble")
        return

    append_job_event(db, job, f"Assembling {len(asset_ids)} clips...")
    try:
        paths: list[Path] = []
        for aid in asset_ids:
            asset = db.get(Asset, aid)
            if asset is None or asset.owner_id != user.id or asset.project_id != project.id:
                raise RuntimeError(f"Clip not found: {aid}")
            if asset.kind != AssetKind.video:
                raise RuntimeError(f"Asset is not a video: {aid}")
            path = Path(asset.file_path)
            if not path.exists():
                raise RuntimeError(f"Missing file for clip: {aid}")
            paths.append(path)

        out_dir = project_storage(user.id, project.id)
        out_path = out_dir / "videos" / f"film_{job.id[:8]}.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        # Re-encode concat so mixed codecs/sizes still join cleanly
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "concat.txt"
            # ffmpeg concat demuxer on Windows needs escaped single quotes carefully;
            # write paths with forward slashes inside single quotes.
            lines = []
            for p in paths:
                escaped = p.resolve().as_posix().replace("'", r"'\''")
                lines.append(f"file '{escaped}'")
            list_file.write_text("\n".join(lines), encoding="utf-8")
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "ffmpeg assemble failed")

        asset = create_asset(
            db,
            project=project,
            user=user,
            kind=AssetKind.video,
            prompt=job.prompt,
            file_path=str(out_path),
            mime_type="video/mp4",
            meta={
                "role": "film",
                "job_id": job.id,
                "source_clips": asset_ids,
                "clip_count": len(asset_ids),
            },
        )
        job.result_asset_ids = json.dumps([asset.id])
        append_job_event(db, job, f"Film saved: {asset.id} ({len(asset_ids)} clips)")
        set_job_status(db, job, JobStatus.succeeded)
    except Exception as exc:
        append_job_event(db, job, f"Failed: {exc}")
        set_job_status(db, job, JobStatus.failed, str(exc))


def run_movie_job(db: Session, job_id: str) -> None:
    job = db.get(Job, job_id)
    if job is None:
        return
    user = db.get(User, job.owner_id)
    project = db.get(Project, job.project_id)
    if user is None or project is None:
        set_job_status(db, job, JobStatus.failed, "Missing user or project")
        return

    set_job_status(db, job, JobStatus.running)
    out_dir = project_storage(user.id, project.id) / "scenes"
    asset_ids: list[str] = []
    video_clip_ids: list[str] = []

    def on_progress(message: str) -> None:
        append_job_event(db, job, message)

    try:
        orchestrator = Orchestrator(output_dir=out_dir)
        packets = orchestrator.run(job.prompt, progress_callback=on_progress)
        for packet in packets:
            if packet.image_path:
                asset = create_asset(
                    db,
                    project=project,
                    user=user,
                    kind=AssetKind.image,
                    prompt=packet.image_prompt or job.prompt,
                    file_path=packet.image_path,
                    mime_type="image/png",
                    meta={"scene": packet.scene_number, "job_id": job.id, "title": packet.title},
                )
                asset_ids.append(asset.id)
            if packet.video_path:
                asset = create_asset(
                    db,
                    project=project,
                    user=user,
                    kind=AssetKind.video,
                    prompt=packet.video_prompt or job.prompt,
                    file_path=packet.video_path,
                    mime_type="video/mp4",
                    meta={
                        "scene": packet.scene_number,
                        "job_id": job.id,
                        "title": packet.title,
                        "role": "clip",
                        "clip_index": packet.scene_number,
                    },
                )
                asset_ids.append(asset.id)
                video_clip_ids.append(asset.id)
            packet_path = out_dir / f"scene_{packet.scene_number}_packet.json"
            if packet_path.exists():
                asset = create_asset(
                    db,
                    project=project,
                    user=user,
                    kind=AssetKind.packet,
                    prompt=job.prompt,
                    file_path=str(packet_path),
                    mime_type="application/json",
                    meta={"scene": packet.scene_number, "job_id": job.id},
                )
                asset_ids.append(asset.id)

        # Settle movie cost to actual scene count if we over/under estimated
        actual_cost = max(1, len(packets)) * settings.credit_cost_movie_scene
        delta = actual_cost - job.credits_charged
        if delta > 0:
            try:
                credit_service.charge_credits(db, user, delta, reason="movie_scene_adjust", job_id=job.id)
                job.credits_charged = actual_cost
            except ValueError:
                append_job_event(db, job, f"Warning: could not charge remaining {delta} credits")
        elif delta < 0:
            credit_service.add_credits(db, user, -delta, reason="movie_scene_refund", job_id=job.id)
            job.credits_charged = actual_cost

        if len(video_clip_ids) >= 2:
            append_job_event(db, job, f"Assembling {len(video_clip_ids)} scene clips into final film...")
            try:
                import subprocess
                import tempfile

                import imageio_ffmpeg

                paths: list[Path] = []
                for aid in video_clip_ids:
                    clip_asset = db.get(Asset, aid)
                    if clip_asset and Path(clip_asset.file_path).exists():
                        paths.append(Path(clip_asset.file_path))
                if len(paths) >= 2:
                    film_path = project_storage(user.id, project.id) / "videos" / f"film_{job.id[:8]}.mp4"
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                    with tempfile.TemporaryDirectory() as tmp:
                        list_file = Path(tmp) / "concat.txt"
                        lines = []
                        for p in paths:
                            escaped = p.resolve().as_posix().replace("'", r"'\''")
                            lines.append(f"file '{escaped}'")
                        list_file.write_text("\n".join(lines), encoding="utf-8")
                        cmd = [
                            ffmpeg,
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            str(list_file),
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-an",
                            str(film_path),
                        ]
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        if proc.returncode == 0:
                            film = create_asset(
                                db,
                                project=project,
                                user=user,
                                kind=AssetKind.video,
                                prompt=job.prompt,
                                file_path=str(film_path),
                                mime_type="video/mp4",
                                meta={
                                    "role": "film",
                                    "job_id": job.id,
                                    "source_clips": video_clip_ids,
                                    "clip_count": len(video_clip_ids),
                                },
                            )
                            asset_ids.append(film.id)
                            append_job_event(db, job, f"Final film saved: {film.id}")
                        else:
                            append_job_event(
                                db, job, f"Assemble skipped: {proc.stderr.strip() or 'ffmpeg error'}"
                            )
            except Exception as assemble_exc:
                append_job_event(db, job, f"Assemble skipped: {assemble_exc}")

        job.result_asset_ids = json.dumps(asset_ids)
        append_job_event(db, job, f"Movie complete with {len(packets)} scenes")
        set_job_status(db, job, JobStatus.succeeded)
    except Exception as exc:
        append_job_event(db, job, f"Failed: {exc}")
        set_job_status(db, job, JobStatus.failed, str(exc))

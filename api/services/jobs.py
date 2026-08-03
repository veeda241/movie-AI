from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Prefer the repo-root video_lab package (Wan2.1 generator) on sys.path.
_file_dir = Path(__file__).resolve().parent
for _candidate in (_file_dir.parents[0], *_file_dir.parents):
    if (_candidate / "video_lab" / "__init__.py").exists():
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

# UI model ids (web/app/app/page.tsx)
WAN_1_3B_MODEL_ID = "wan-2.1-1.3b"
WAN_2_2_MODEL_ID = "wan-2.2"
MOTIF_LOCAL_MODEL_ID = "motif-local"

WAN_ALLOW_PATTERNS = [
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
    "scheduler/*",
    "model_index.json",
]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


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
    """Run local Wan2.1-T2V-1.3B (`generate_finetune_video`) into `out_path`."""
    import shutil

    def _log(msg: str) -> None:
        try:
            append_job_event(db, job, msg)
        except Exception:
            pass

    _log("Pre-flight: checking CUDA…")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. The local Wan 2.1 1.3B model requires an NVIDIA GPU. "
                "Pick wan-2.2 (remote) or motif-local, or run on a CUDA host."
            )
        _log(f"Pre-flight: CUDA ok ({torch.cuda.get_device_name(0)})")
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install requirements-model.txt to use the local Wan model."
        ) from exc

    from video_lab.config import LabConfig
    from video_lab.infer.finetune_generate import generate_finetune_video

    cfg = LabConfig()
    base = cfg.base_t2v_model

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        no_exist = Path(HF_HUB_CACHE) / ("models--" + base.replace("/", "--")) / ".no_exist"
        if no_exist.exists():
            shutil.rmtree(no_exist, ignore_errors=True)
            _log(f"Cleared stale .no_exist cache at {no_exist}")
    except Exception as cache_exc:
        _log(f"Cache cleanup skipped: {cache_exc}")

    try:
        from huggingface_hub import snapshot_download

        last_event = {"t": 0.0}

        def _tqdm(msg_callback):
            try:
                from tqdm.auto import tqdm as _tqdm_lib
            except ImportError:
                return None

            class _Wrap(_tqdm_lib):
                def update(self, n=1.0):  # type: ignore[override]
                    res = super().update(n)
                    now = time.time()
                    # Throttle DB spam: at most one progress event every 3s.
                    if now - last_event["t"] >= 3.0 or self.n >= (self.total or 0):
                        last_event["t"] = now
                        msg_callback(
                            f"Downloading {self.desc or 'file'}: "
                            f"{self.n:,}/{self.total or 0:,} bytes "
                            f"({100 * self.n / max(self.total or 1, 1):.0f}%)"
                        )
                    return res

            return _Wrap

        _log(f"Pre-fetching Wan model: {base} (~12 GB on first run)")
        snapshot_download(
            repo_id=base,
            allow_patterns=WAN_ALLOW_PATTERNS,
            tqdm_class=_tqdm(_log),
            max_workers=2,
        )
        _log("Wan model files present in cache")
    except Exception as dl_exc:
        raise RuntimeError(
            f"Failed to download Wan model {base!r}: {dl_exc}. "
            f"Run `python scripts/download_wan.py --clear-no-exist` to retry manually."
        ) from dl_exc

    seed = int.from_bytes(job.id.encode("utf-8")[:8], "big") % (2**32)
    use_lora = _env_bool("WAN_USE_LORA", False)

    _log("Building Wan pipeline (loading weights into VRAM)…")
    if use_lora:
        _log("WAN_USE_LORA=true — will load LoRA adapter if present")

    wan_path = generate_finetune_video(
        job.prompt,
        seed=seed,
        steps=int(os.environ.get("WAN_STEPS", "30")),
        # 81 frames is native Wan but heavy on 16GB; 33 ≈ 2s @16fps and is 4N+1.
        frames=int(os.environ.get("WAN_FRAMES", "33")),
        fps=int(os.environ.get("WAN_FPS", "16")),
        height=int(os.environ.get("WAN_HEIGHT", "480")),
        width=int(os.environ.get("WAN_WIDTH", "832")),
        use_lora=use_lora,
        log_fn=_log,
    )

    if not wan_path or not Path(wan_path).exists():
        raise RuntimeError(f"Wan generator returned no file: {wan_path!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wan_path, out_path)
    return str(out_path)


def _generate_with_motif(job: Job, out_dir: Path, out_path: Path, clip_index: int) -> str:
    """Route UI model ids to MotifClient presets."""
    model = (job.model or "").strip().lower()
    if model == MOTIF_LOCAL_MODEL_ID:
        client = MotifClient(output_dir=out_dir / "videos", force_local=True, ui_model=model)
    elif model == WAN_2_2_MODEL_ID:
        client = MotifClient(output_dir=out_dir / "videos", ui_model=WAN_2_2_MODEL_ID)
    else:
        client = MotifClient(output_dir=out_dir / "videos", ui_model=model or None)
    return client.generate(job.prompt, scene_number=clip_index, output_path=out_path)


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
            path = _generate_with_motif(job, out_dir, out_path, clip_index)
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
    if not os.environ.get("HF_TOKEN", "").strip():
        append_job_event(
            db,
            job,
            "HF_TOKEN not set — using local planning (3 deterministic scenes) "
            "and the local 'placeholder' video fallback. Set HF_TOKEN in .env "
            "for LLM-driven planning and real video generation.",
        )
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

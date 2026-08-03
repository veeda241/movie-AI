"""Movie Flow API — FastAPI SaaS backend wrapping movie_pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load repo-root .env so HF_TOKEN and other secrets reach agents/clients
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env", override=False)

# Windows Downloads folders often block `_regex.pyd`; shim before transformers/diffusers.
try:
    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from video_lab.utils.regex_shim import ensure_regex_shim

    ensure_regex_shim()
except Exception:
    pass

from api.config import settings
from api.db import Base, engine, SessionLocal
from api.routers import auth, billing, generate, jobs, orgs, projects, assets
from api.services.bootstrap import ensure_default_plans

# Ensure storage root exists
Path(settings.storage_root).mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(bind=engine)


def _print_startup_banner() -> None:
    """One-shot diagnostic banner so operators can spot misconfiguration
    before users hit the API."""
    if os.environ.get("API_DIAGNOSTICS", "true").strip().lower() in {"0", "false", "no", "off"}:
        return

    hf_token_set = bool(os.environ.get("HF_TOKEN", "").strip())
    hf_video_model = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
    hf_video_provider = os.environ.get("HF_VIDEO_PROVIDER", "hf-inference")
    hf_text_model = os.environ.get("HF_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    allow_fallback = os.environ.get("HF_ALLOW_LOCAL_FALLBACK", "true").strip().lower() not in {"0", "false", "no", "off"}
    video_backend = os.environ.get("VIDEO_BACKEND", "remote")

    cuda_status = "unavailable"
    try:
        import torch

        cuda_status = "available" if torch.cuda.is_available() else "unavailable"
    except Exception:
        cuda_status = "torch-not-installed"

    ffmpeg_status = "ok" if shutil.which("ffmpeg") or _imageio_ffmpeg_ok() else "missing"

    lines = [
        "",
        "================ Movie Flow API ================",
        f"  Database       : {settings.database_url}",
        f"  Storage root   : {settings.storage_root}",
        f"  HF_TOKEN       : {'set' if hf_token_set else 'MISSING — movie pipeline will use local planning; remote video will fail'}",
        f"  HF text model  : {hf_text_model}",
        f"  HF video model : {hf_video_model} (provider={hf_video_provider})",
        f"  VIDEO_BACKEND  : {video_backend} (remote | local-wan | local-placeholder)",
        f"  Local fallback : {'enabled (placeholder videos when remote fails)' if allow_fallback else 'DISABLED (remote failures will surface as job errors)'}",
        f"  CUDA           : {cuda_status} (Wan 2.1 1.3B local model requires CUDA)",
        f"  ffmpeg         : {ffmpeg_status}",
        "=================================================",
        "",
    ]
    print("\n".join(lines), flush=True)


def _imageio_ffmpeg_ok() -> bool:
    try:
        import imageio_ffmpeg

        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def _ensure_sqlite_columns() -> None:
    """Add columns introduced after first create_all (SQLite)."""
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(jobs)").fetchall()
        names = {row[1] for row in rows}
        if "payload_json" not in names:
            conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN payload_json TEXT DEFAULT '{}'")


_ensure_sqlite_columns()

app = FastAPI(title="Movie Flow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(assets.router, prefix="/assets", tags=["assets"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(billing.router, prefix="/billing", tags=["billing"])
app.include_router(billing.webhook_router, prefix="/webhooks", tags=["webhooks"])
app.include_router(orgs.router, prefix="/orgs", tags=["orgs"])


@app.on_event("startup")
def on_startup() -> None:
    _print_startup_banner()
    db = SessionLocal()
    try:
        ensure_default_plans(db)
    finally:
        db.close()


@app.get("/health")
def health() -> dict:
    """Liveness probe + diagnostic snapshot so the UI / load balancer can tell
    whether the API has HF_TOKEN and CUDA wired up correctly."""
    snapshot: dict = {"status": "ok", "product": "Movie Flow"}
    snapshot["hf_token"] = "set" if os.environ.get("HF_TOKEN", "").strip() else "missing"
    snapshot["hf_video_model"] = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
    snapshot["hf_video_provider"] = os.environ.get("HF_VIDEO_PROVIDER", "hf-inference")
    snapshot["video_backend"] = os.environ.get("VIDEO_BACKEND", "remote")
    snapshot["local_fallback"] = os.environ.get("HF_ALLOW_LOCAL_FALLBACK", "true")
    try:
        import torch

        snapshot["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        snapshot["cuda"] = False
    snapshot["ffmpeg"] = _imageio_ffmpeg_ok() or bool(shutil.which("ffmpeg"))
    return snapshot

"""Movie Flow API — FastAPI SaaS backend wrapping movie_pipeline."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load repo-root .env so HF_TOKEN and other secrets reach agents/clients
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from api.config import settings
from api.db import Base, engine, SessionLocal
from api.routers import auth, billing, generate, jobs, orgs, projects, assets
from api.services.bootstrap import ensure_default_plans

# Ensure storage root exists
Path(settings.storage_root).mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(bind=engine)


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
    db = SessionLocal()
    try:
        ensure_default_plans(db)
    finally:
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "Movie Flow"}

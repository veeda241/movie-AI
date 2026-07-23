from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth_utils import get_current_user
from api.config import settings
from api.db import SessionLocal, get_db
from api.models import JobKind, Project, User
from api.schemas import (
    AssembleRequest,
    GenerateImageRequest,
    GenerateMovieRequest,
    GenerateVideoRequest,
    JobOut,
)
from api.services import jobs as job_service

router = APIRouter()


def _job_out(job) -> JobOut:
    return JobOut(
        id=job.id,
        project_id=job.project_id,
        kind=job.kind,
        status=job.status,
        prompt=job.prompt,
        model=job.model,
        credits_charged=job.credits_charged,
        result_asset_ids=json.loads(job.result_asset_ids or "[]"),
        error=job.error or "",
        events=json.loads(job.events_json or "[]"),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _get_project(db: Session, user: User, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None or project.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _run_in_bg(runner, job_id: str) -> None:
    db = SessionLocal()
    try:
        runner(db, job_id)
    finally:
        db.close()


@router.post("/image", response_model=JobOut)
def generate_image(
    body: GenerateImageRequest,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    project = _get_project(db, user, body.project_id)
    cost = settings.credit_cost_image
    try:
        job = job_service.create_job(
            db,
            user=user,
            project=project,
            kind=JobKind.image,
            prompt=body.prompt,
            model=body.model,
            credits_charged=cost,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)) from exc
    background.add_task(_run_in_bg, job_service.run_image_job, job.id)
    return _job_out(job)


@router.post("/video", response_model=JobOut)
def generate_video(
    body: GenerateVideoRequest,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    project = _get_project(db, user, body.project_id)
    cost = settings.credit_cost_video
    try:
        job = job_service.create_job(
            db,
            user=user,
            project=project,
            kind=JobKind.video,
            prompt=body.prompt,
            model=body.model,
            credits_charged=cost,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)) from exc
    background.add_task(_run_in_bg, job_service.run_video_job, job.id)
    return _job_out(job)


@router.post("/movie", response_model=JobOut)
def generate_movie(
    body: GenerateMovieRequest,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    project = _get_project(db, user, body.project_id)
    cost = job_service.estimate_movie_cost(3)
    try:
        job = job_service.create_job(
            db,
            user=user,
            project=project,
            kind=JobKind.movie,
            prompt=body.prompt,
            model=body.model,
            credits_charged=cost,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)) from exc
    background.add_task(_run_in_bg, job_service.run_movie_job, job.id)
    return _job_out(job)


@router.post("/assemble", response_model=JobOut)
def assemble_clips(
    body: AssembleRequest,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    project = _get_project(db, user, body.project_id)
    if len(body.asset_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 clips to assemble")
    cost = settings.credit_cost_assemble
    try:
        job = job_service.create_job(
            db,
            user=user,
            project=project,
            kind=JobKind.assemble,
            prompt=body.title.strip() or "Assembled film",
            model="ffmpeg-concat",
            credits_charged=cost,
            payload={"asset_ids": body.asset_ids},
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)) from exc
    background.add_task(_run_in_bg, job_service.run_assemble_job, job.id)
    return _job_out(job)

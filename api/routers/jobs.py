from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.auth_utils import get_current_user
from api.db import SessionLocal, get_db
from api.models import Job, JobStatus, User
from api.schemas import JobOut

router = APIRouter()


def _job_out(job: Job) -> JobOut:
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


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    job = db.get(Job, job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _job_out(job)


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None or job.owner_id != user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    finally:
        db.close()

    async def event_stream():
        last_count = 0
        while True:
            db = SessionLocal()
            try:
                job = db.get(Job, job_id)
                if job is None:
                    yield f"data: {json.dumps({'error': 'missing'})}\n\n"
                    break
                events = json.loads(job.events_json or "[]")
                if len(events) > last_count:
                    for message in events[last_count:]:
                        yield f"data: {json.dumps({'event': message, 'status': job.status.value})}\n\n"
                    last_count = len(events)
                if job.status in {JobStatus.succeeded, JobStatus.failed}:
                    yield f"data: {json.dumps({'done': True, 'status': job.status.value, 'job': _job_out(job).model_dump(mode='json')})}\n\n"
                    break
            finally:
                db.close()
            await asyncio.sleep(0.8)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

"""Run long GPU / movie jobs off the FastAPI event loop.

Starlette ``BackgroundTasks`` execute sync callables *on the event loop thread*
after the response. Loading Wan into VRAM blocks that loop for minutes, so
``GET /jobs/{id}`` polls fail with browser ``Failed to fetch``.

Heavy runners are submitted to a dedicated thread pool instead.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="movie-flow-job")


def submit_job(runner: Callable, job_id: str) -> None:
    """Fire-and-forget job runner on a worker thread."""
    _executor.submit(_run_safe, runner, job_id)


def _run_safe(runner: Callable, job_id: str) -> None:
    from api.db import SessionLocal
    from api.models import Job, JobStatus
    from api.services.jobs import append_job_event, set_job_status

    db = SessionLocal()
    try:
        runner(db, job_id)
    except Exception as exc:
        try:
            job = db.get(Job, job_id)
            if job is not None and job.status in {JobStatus.queued, JobStatus.running}:
                append_job_event(db, job, f"Worker crashed before completion: {exc}")
                set_job_status(
                    db,
                    job,
                    JobStatus.failed,
                    f"{exc}\n{traceback.format_exc()[:1500]}",
                )
        except Exception:
            pass
        traceback.print_exc()
    finally:
        db.close()

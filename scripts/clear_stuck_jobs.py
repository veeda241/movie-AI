"""One-shot helper to mark stuck 'running' jobs as failed.

Use when the API worker has wedged (e.g. blocked inside a model download
with no progress events) and the UI is stuck on 'Working...'.

    python scripts/clear_stuck_jobs.py
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path

DB = Path("storage/movie_flow.db")


def main() -> int:
    if not DB.exists():
        print(f"DB not found at {DB}")
        return 1
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    reason = (
        "Worker was stuck (no progress events for >1h). Likely blocked inside a "
        "huggingface_hub download with stale .no_exist cache entries. Run "
        "scripts/download_wan.py to pre-fetch the Wan model, then re-submit the job."
    )
    cur.execute(
        "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status='running'",
        (reason, now),
    )
    n = cur.rowcount
    con.commit()
    con.close()
    print(f"Marked {n} stuck job(s) as failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

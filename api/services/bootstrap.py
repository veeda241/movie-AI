from __future__ import annotations

# Placeholder for plan catalog seeding (plans are code-defined in billing router).

from sqlalchemy.orm import Session


def ensure_default_plans(db: Session) -> None:
    _ = db  # no DB rows required; plans are static

from __future__ import annotations

from sqlalchemy.orm import Session

from api.config import settings
from api.models import CreditLedger, PlanTier, User


PLAN_CREDITS = {
    PlanTier.free: settings.free_signup_credits,
    PlanTier.starter: settings.starter_monthly_credits,
    PlanTier.pro: settings.pro_monthly_credits,
    PlanTier.enterprise: settings.pro_monthly_credits * 5,
}


def add_credits(db: Session, user: User, delta: int, reason: str, job_id: str | None = None) -> User:
    user.credit_balance = int(user.credit_balance) + int(delta)
    db.add(
        CreditLedger(
            user_id=user.id,
            delta=delta,
            reason=reason,
            job_id=job_id,
        )
    )
    db.commit()
    db.refresh(user)
    return user


def ensure_credits(user: User, cost: int) -> None:
    if user.credit_balance < cost:
        raise ValueError(f"Insufficient credits: need {cost}, have {user.credit_balance}")


def charge_credits(db: Session, user: User, cost: int, reason: str, job_id: str | None = None) -> User:
    ensure_credits(user, cost)
    return add_credits(db, user, -cost, reason, job_id=job_id)


def grant_plan_credits(db: Session, user: User, plan: PlanTier) -> User:
    amount = PLAN_CREDITS.get(plan, settings.free_signup_credits)
    user.plan = plan
    return add_credits(db, user, amount, f"plan_grant:{plan.value}")

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.auth_utils import get_current_user
from api.config import settings
from api.db import get_db
from api.models import PlanTier, Subscription, User
from api.schemas import CheckoutRequest, PlanOut
from api.services import credits as credit_service

router = APIRouter()
webhook_router = APIRouter()

PLANS = [
    PlanOut(
        id="starter",
        name="Team Starter",
        price_monthly=49,
        credits=settings.starter_monthly_credits,
        seats=5,
        features=[
            "Multi-agent planning",
            "Local fallback rendering",
            "Shared dashboard",
            f"{settings.starter_monthly_credits} credits / month",
        ],
    ),
    PlanOut(
        id="pro",
        name="Professional Studio",
        price_monthly=149,
        credits=settings.pro_monthly_credits,
        seats=20,
        features=[
            "Character consistency tracking",
            "Style references",
            "Asset export to editors",
            f"{settings.pro_monthly_credits} credits / month",
        ],
    ),
    PlanOut(
        id="enterprise",
        name="Enterprise",
        price_monthly=0,
        credits=0,
        seats=100,
        features=["Custom quotas", "SAML SSO", "Dedicated support", "Custom LoRA training"],
    ),
]


def _stripe():
    if not settings.stripe_secret_key:
        return None
    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe


@router.get("/plans", response_model=list[PlanOut])
def list_plans() -> list[PlanOut]:
    return PLANS


@router.post("/checkout")
def create_checkout(
    body: CheckoutRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    stripe = _stripe()
    if stripe is None:
        # Local demo mode: apply plan/credits without Stripe
        if body.plan in {"starter", "pro"}:
            plan = PlanTier.starter if body.plan == "starter" else PlanTier.pro
            credit_service.grant_plan_credits(db, user, plan)
            return {"mode": "demo", "status": "upgraded", "plan": plan.value, "credits": user.credit_balance}
        if body.credit_pack:
            credit_service.add_credits(db, user, body.credit_pack, "demo_topup")
            return {"mode": "demo", "status": "topped_up", "credits": user.credit_balance}
        raise HTTPException(status_code=400, detail="Specify plan or credit_pack")

    if not user.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        user.stripe_customer_id = customer["id"]
        db.add(user)
        db.commit()

    line_items = []
    mode = "subscription"
    metadata = {"user_id": user.id}

    if body.plan == "starter" and settings.stripe_price_starter:
        line_items = [{"price": settings.stripe_price_starter, "quantity": 1}]
        metadata["plan"] = "starter"
    elif body.plan == "pro" and settings.stripe_price_pro:
        line_items = [{"price": settings.stripe_price_pro, "quantity": 1}]
        metadata["plan"] = "pro"
    elif body.credit_pack == 100 and settings.stripe_price_credits_100:
        line_items = [{"price": settings.stripe_price_credits_100, "quantity": 1}]
        mode = "payment"
        metadata["credit_pack"] = "100"
    elif body.credit_pack == 500 and settings.stripe_price_credits_500:
        line_items = [{"price": settings.stripe_price_credits_500, "quantity": 1}]
        mode = "payment"
        metadata["credit_pack"] = "500"
    else:
        raise HTTPException(status_code=400, detail="Stripe price not configured for this option")

    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        mode=mode,
        line_items=line_items,
        success_url=f"{settings.frontend_url}/app/settings/billing?success=1",
        cancel_url=f"{settings.frontend_url}/pricing?canceled=1",
        metadata=metadata,
    )
    return {"mode": "stripe", "url": session.url}


@router.post("/portal")
def billing_portal(user: User = Depends(get_current_user)) -> dict:
    stripe = _stripe()
    if stripe is None or not user.stripe_customer_id:
        return {"mode": "demo", "url": f"{settings.frontend_url}/app/settings/billing"}
    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=f"{settings.frontend_url}/app/settings/billing",
    )
    return {"mode": "stripe", "url": session.url}


@webhook_router.post("/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)) -> dict[str, str]:
    stripe = _stripe()
    if stripe is None:
        return {"status": "stripe_disabled"}

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        if settings.stripe_webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig, settings.stripe_webhook_secret)
        else:
            event = stripe.Event.construct_from(json_loads_safe(payload), stripe.api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Webhook error: {exc}") from exc

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        meta = session.get("metadata") or {}
        user_id = meta.get("user_id")
        user = db.get(User, user_id) if user_id else None
        if user:
            if meta.get("plan") == "starter":
                credit_service.grant_plan_credits(db, user, PlanTier.starter)
            elif meta.get("plan") == "pro":
                credit_service.grant_plan_credits(db, user, PlanTier.pro)
            elif meta.get("credit_pack"):
                credit_service.add_credits(db, user, int(meta["credit_pack"]), "stripe_topup")
            sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
            if sub is None:
                sub = Subscription(user_id=user.id)
            sub.plan = user.plan
            sub.status = "active"
            sub.stripe_subscription_id = session.get("subscription")
            db.add(sub)
            db.commit()

    return {"status": "ok"}


def json_loads_safe(payload: bytes):
    import json

    return json.loads(payload.decode("utf-8"))

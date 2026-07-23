from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth_utils import get_current_user
from api.db import get_db
from api.models import Membership, MembershipRole, Org, OrgInvite, PlanTier, User
from api.schemas import AcceptInviteRequest, InviteOut, InviteRequest, MemberOut, OrgCreate, OrgOut

router = APIRouter()

SEAT_LIMITS = {
    PlanTier.free: 1,
    PlanTier.starter: 5,
    PlanTier.pro: 20,
    PlanTier.enterprise: 100,
}


def _org_out(org: Org, db: Session) -> OrgOut:
    count = db.query(Membership).filter(Membership.org_id == org.id).count()
    return OrgOut(
        id=org.id,
        name=org.name,
        plan=org.plan,
        seat_limit=org.seat_limit,
        member_count=count,
    )


@router.get("", response_model=list[OrgOut])
def list_orgs(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[OrgOut]:
    memberships = db.query(Membership).filter(Membership.user_id == user.id).all()
    orgs = []
    for m in memberships:
        org = db.get(Org, m.org_id)
        if org:
            orgs.append(_org_out(org, db))
    return orgs


@router.post("", response_model=OrgOut)
def create_org(
    body: OrgCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrgOut:
    plan = user.plan if user.plan != PlanTier.free else PlanTier.starter
    org = Org(name=body.name, plan=plan, seat_limit=SEAT_LIMITS.get(plan, 5))
    db.add(org)
    db.commit()
    db.refresh(org)
    membership = Membership(org_id=org.id, user_id=user.id, role=MembershipRole.owner)
    db.add(membership)
    db.commit()
    return _org_out(org, db)


@router.get("/{org_id}/members", response_model=list[MemberOut])
def list_members(
    org_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MemberOut]:
    _require_member(db, user, org_id)
    rows = db.query(Membership).filter(Membership.org_id == org_id).all()
    out = []
    for row in rows:
        member = db.get(User, row.user_id)
        if member:
            out.append(MemberOut(user_id=member.id, email=member.email, name=member.name, role=row.role))
    return out


@router.post("/{org_id}/invite", response_model=InviteOut)
def invite_member(
    org_id: str,
    body: InviteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrgInvite:
    membership = _require_member(db, user, org_id)
    if membership.role not in {MembershipRole.owner, MembershipRole.admin}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owners/admins can invite")
    org = db.get(Org, org_id)
    assert org is not None
    current = db.query(Membership).filter(Membership.org_id == org_id).count()
    pending = db.query(OrgInvite).filter(OrgInvite.org_id == org_id, OrgInvite.accepted.is_(False)).count()
    if current + pending >= org.seat_limit:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Seat limit reached")
    invite = OrgInvite(org_id=org_id, email=body.email.lower(), role=body.role)
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite


@router.post("/accept-invite")
def accept_invite(
    body: AcceptInviteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    invite = db.query(OrgInvite).filter(OrgInvite.token == body.token, OrgInvite.accepted.is_(False)).first()
    if invite is None:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.email != user.email.lower():
        raise HTTPException(status_code=403, detail="Invite email mismatch")
    existing = (
        db.query(Membership)
        .filter(Membership.org_id == invite.org_id, Membership.user_id == user.id)
        .first()
    )
    if existing is None:
        db.add(Membership(org_id=invite.org_id, user_id=user.id, role=invite.role))
    invite.accepted = True
    db.add(invite)
    db.commit()
    return {"status": "accepted", "org_id": invite.org_id}


def _require_member(db: Session, user: User, org_id: str) -> Membership:
    membership = (
        db.query(Membership)
        .filter(Membership.org_id == org_id, Membership.user_id == user.id)
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Org not found")
    return membership

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from api.models import AssetKind, JobKind, JobStatus, MembershipRole, PlanTier


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    plan: PlanTier
    credit_balance: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = ""
    org_id: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    org_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AssetOut(BaseModel):
    id: str
    project_id: str
    kind: AssetKind
    prompt: str
    mime_type: str
    created_at: datetime
    file_url: str | None = None
    meta: dict[str, Any] = {}

    model_config = {"from_attributes": True}


class GenerateImageRequest(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1)
    model: str = "sdxl-local"


class GenerateVideoRequest(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1)
    model: str = "wan-2.2"
    start_frame_asset_id: str | None = None


class GenerateMovieRequest(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1)
    model: str = "multi-agent"


class AssembleRequest(BaseModel):
    project_id: str
    asset_ids: list[str] = Field(min_length=2)
    title: str = "Assembled film"


class JobOut(BaseModel):
    id: str
    project_id: str
    kind: JobKind
    status: JobStatus
    prompt: str
    model: str
    credits_charged: int
    result_asset_ids: list[str] = []
    error: str = ""
    events: list[str] = []
    created_at: datetime
    updated_at: datetime


class PlanOut(BaseModel):
    id: str
    name: str
    price_monthly: int
    credits: int
    seats: int
    features: list[str]


class CheckoutRequest(BaseModel):
    plan: str | None = None
    credit_pack: int | None = None


class OrgCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class OrgOut(BaseModel):
    id: str
    name: str
    plan: PlanTier
    seat_limit: int
    member_count: int = 0

    model_config = {"from_attributes": True}


class InviteRequest(BaseModel):
    email: EmailStr
    role: MembershipRole = MembershipRole.member


class InviteOut(BaseModel):
    id: str
    email: str
    role: MembershipRole
    token: str
    accepted: bool

    model_config = {"from_attributes": True}


class MemberOut(BaseModel):
    user_id: str
    email: str
    name: str
    role: MembershipRole


class AcceptInviteRequest(BaseModel):
    token: str

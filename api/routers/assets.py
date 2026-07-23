from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from api.auth_utils import get_current_user, get_current_user_flexible
from api.db import get_db
from api.models import Asset, AssetKind, Project, User
from api.schemas import AssetOut

router = APIRouter()


def _asset_out(asset: Asset) -> AssetOut:
    try:
        meta = json.loads(asset.meta_json or "{}")
    except json.JSONDecodeError:
        meta = {}
    return AssetOut(
        id=asset.id,
        project_id=asset.project_id,
        kind=asset.kind,
        prompt=asset.prompt,
        mime_type=asset.mime_type,
        created_at=asset.created_at,
        file_url=f"/assets/{asset.id}/file",
        meta=meta if isinstance(meta, dict) else {},
    )


@router.get("/export/project/{project_id}")
def export_project_zip(
    project_id: str,
    user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    project = db.get(Project, project_id)
    if project is None or project.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    assets = db.query(Asset).filter(Asset.project_id == project_id, Asset.owner_id == user.id).all()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for asset in assets:
            path = Path(asset.file_path)
            if path.exists():
                arcname = f"{asset.kind.value}/{path.name}"
                zf.write(path, arcname=arcname)
                manifest.append(
                    {
                        "id": asset.id,
                        "kind": asset.kind.value,
                        "prompt": asset.prompt,
                        "file": arcname,
                    }
                )
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{project.name}.zip"'},
    )


@router.get("", response_model=list[AssetOut])
def list_assets(
    project_id: str | None = None,
    kind: AssetKind | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AssetOut]:
    query = db.query(Asset).filter(Asset.owner_id == user.id)
    if project_id:
        query = query.filter(Asset.project_id == project_id)
    if kind:
        query = query.filter(Asset.kind == kind)
    assets = query.order_by(Asset.created_at.desc()).all()
    return [_asset_out(a) for a in assets]


@router.get("/{asset_id}", response_model=AssetOut)
def get_asset(
    asset_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AssetOut:
    asset = db.get(Asset, asset_id)
    if asset is None or asset.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return _asset_out(asset)


@router.get("/{asset_id}/file")
def get_asset_file(
    asset_id: str,
    user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
) -> FileResponse:
    asset = db.get(Asset, asset_id)
    if asset is None or asset.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    path = Path(asset.file_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File missing on disk")
    return FileResponse(path, media_type=asset.mime_type, filename=path.name)

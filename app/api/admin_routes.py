"""Admin API — model + data lifecycle on the persistent volume.

Every route is guarded by `require_admin` (API key + IP allowlist + rate-limit
+ audit). Mounted under /admin in app.main.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Path
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import require_admin
from app.services import admin_service, predict_service

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)


class SyncS3Request(BaseModel):
    version_ids: Optional[list[str]] = Field(
        None, description="S3 version ids to pull, e.g. ['v4']. Empty/omitted = ALL versions in the S3 registry."
    )
    sync_registry: bool = Field(True, description="Merge S3 registry entries into volume registry.json")


@router.get("/health", summary="Volume + model/data status")
def admin_health():
    return admin_service.health()


@router.get("/models", summary="List model versions on the volume")
def admin_list_models():
    return admin_service.get_models_status()


@router.post("/models/sync-s3", summary="Pull model version(s) from the S3 bucket onto the volume")
def admin_sync_s3(body: SyncS3Request):
    try:
        result = admin_service.sync_s3(body.version_ids, sync_registry=body.sync_registry)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    status = 207 if result["errors"] else 200
    return {"status": status, **result}


@router.delete("/models/{version_id}", summary="Delete a model version from the volume")
def admin_delete_model(version_id: str = Path(..., description="Version id to delete, e.g. 'v2'")):
    try:
        return admin_service.delete_model(version_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/models/{version_id}/preload", summary="Load a model into memory now")
def admin_preload_model(version_id: str = Path(..., description="Version id to preload")):
    if not settings.MODEL_CACHE_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="Model caching is disabled (MODEL_CACHE_ENABLED=False); "
                   "models load fresh per request and are not retained.",
        )
    try:
        predict_service.load_model_to_memory(version_id)
        return {"loaded": version_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/data/upload", summary="Upload a prebuilt cars.duckdb (atomic swap)")
async def admin_upload_data(file: UploadFile = File(..., description="cars.duckdb built locally")):
    try:
        return admin_service.upload_data(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid cars.duckdb: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

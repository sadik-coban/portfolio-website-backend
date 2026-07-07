import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router
from app.api.admin_routes import router as admin_router
from app.services.predict_service import preload_latest_model, unload_models
from app.services import data_sync_service
from app.services import lgb_service

os.makedirs(settings.STATIC_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting application... model preload from volume ({settings.VOLUME_DIR}).")
    try:
        preload_latest_model()
        print("Latest model loaded into memory from volume and ready.")
    except Exception as e:
        # No registry/model on the volume yet is fine — admin can populate it,
        # and models load lazily on first /predict.
        print(f"No model preloaded at startup: {e}")
        print("Model will be loaded on first request (after admin sync).")

    # Price-prediction serving model (LightGBM · TF-IDF+SVD): download from S3 → memory.
    try:
        lgb_service.load_model()
        print(f"Serving model loaded into memory from S3 ({settings.SERVING_MODEL_KEY}).")
    except Exception as e:
        print(f"Serving model not preloaded ({e}); will load lazily on first /api/predict.")

    # One-time data bootstrap: pull cars.duckdb from S3 onto the {VOLUME_DIR} volume if
    # it isn't there yet (Railway mounts an empty /data volume on first boot). This is a
    # single download on startup — NOT the poll loop.
    try:
        from app.core.db import db_exists
        from app.core.s3_client import get_s3_client
        if db_exists():
            print(f"cars.duckdb already on the volume ({settings.DUCKDB_PATH}).")
        else:
            print(f"cars.duckdb missing — downloading once from S3 ({settings.DATA_S3_KEY}) → {settings.DUCKDB_PATH}...")
            os.makedirs(os.path.dirname(settings.DUCKDB_PATH) or ".", exist_ok=True)
            tmp = settings.DUCKDB_PATH + ".tmp"
            with open(tmp, "wb") as f:
                get_s3_client().download_fileobj(settings.RAILWAY_S3_BUCKET, settings.DATA_S3_KEY, f)
            os.replace(tmp, settings.DUCKDB_PATH)  # atomic (same filesystem as the volume)
            print("cars.duckdb downloaded to the volume.")
    except Exception as e:
        print(f"Data bootstrap failed ({e}); dashboard/drift need cars.duckdb on the volume.")

    # Data-sync poll DEACTIVATED for now (2026-07-07). Re-enable by uncommenting below
    # and setting DATA_SYNC_POLL_SECONDS > 0.
    # data_sync_task = None
    # if settings.DATA_SYNC_POLL_SECONDS > 0:
    #     print(f"Starting data-sync poll loop (every {settings.DATA_SYNC_POLL_SECONDS}s).")
    #     data_sync_task = asyncio.create_task(data_sync_service.poll_loop())
    print("Data-sync poll deactivated.")

    yield

    # if data_sync_task is not None:
    #     data_sync_task.cancel()
    #     try:
    #         await data_sync_task
    #     except asyncio.CancelledError:
    #         pass

    print("Shutting down application. Clearing memory...")
    unload_models()

tags_metadata = [
    {
        "name": "Model Management",
        "description": "Operations related to model versions and SHAP explainability.",
    },
    {
        "name": "Prediction",
        "description": "Car price prediction using trained CatBoost quantile regression models.",
    },
    {
        "name": "Drift Detection",
        "description": "Data drift analysis between two model training datasets using KS-test and EMD.",
    },
    {
        "name": "Dashboard & Analytics",
        "description": "Aggregated dashboard data and dropdown options for the frontend.",
    },
    {
        "name": "Admin",
        "description": "Secured model/data lifecycle on the volume (API key + IP allowlist required).",
    },
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "## Car Price Prediction & MLOps API\n\n"
        "A production-grade ML API for **second-hand car price prediction** with built-in MLOps capabilities.\n\n"
        "### Features\n"
        "- 🚗 **Price Prediction** — CatBoost multi-quantile regression (Q5, Q50, Q95)\n"
        "- 📊 **Data Drift Detection** — KS-test & Earth Mover's Distance between training versions\n"
        "- 📈 **Dashboard Analytics** — Aggregated charts, KPIs, and filter-based exploration\n"
        "- 🔍 **SHAP Explainability** — Feature importance visualizations per model version\n"
        "- 🗂️ **Multi-Version Model Registry** — Load, compare, and manage model versions\n"
        "- ☁️ **S3 Storage** — All data served from Railway S3-compatible bucket\n"
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
# Admin routes DEACTIVATED for now (2026-07-07) — /admin/* not mounted (kept in code).
# Re-enable by uncommenting:
# app.include_router(admin_router)

app.mount("/reports", StaticFiles(directory=settings.STATIC_DIR), name="reports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

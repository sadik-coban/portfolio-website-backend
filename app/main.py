import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router
from app.services import bi_service, data_service, lgb_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting application (volume: {settings.VOLUME_DIR}).")

    # Price-prediction serving model (LightGBM · TF-IDF+SVD): download from S3 → memory.
    try:
        lgb_service.load_model()
        print(f"Serving model loaded into memory from S3 ({settings.SERVING_MODEL_KEY}).")
    except Exception as e:
        print(f"Serving model not preloaded ({e}); will load lazily on first /api/predict.")

    # One-time data bootstrap: pull cars.duckdb from S3 onto the {VOLUME_DIR} volume if
    # it isn't there yet (Railway mounts an empty /data volume on first boot). Single
    # download on startup — NOT a poll loop.
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

    # Warm the in-memory caches so the FIRST user request is served from memory:
    #  • bi_service.load_rows() — the 30K-row window-dedup build + columnar arrays
    #  • data_service.get_snapshots() — the snapshot-date list (1 + 2N COUNT queries)
    try:
        bi_service.load_rows()
        data_service.get_snapshots()
        print("BI rows + snapshots warmed into memory.")
    except Exception as e:
        print(f"Warm-up skipped ({e}); caches will build lazily on first request.")

    yield

    print("Shutting down. Clearing model from memory...")
    lgb_service.unload_model()


tags_metadata = [
    {"name": "Prediction", "description": "Used-car price prediction (LightGBM · TF-IDF+SVD)."},
    {"name": "Drift Detection", "description": "Distribution drift between two listing snapshots (KS-test + EMD)."},
    {"name": "Dashboard & Analytics", "description": "BI dashboard aggregation + snapshot dates."},
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "## Car Price Prediction & Analytics API\n\n"
        "- 🚗 **Price Prediction** — LightGBM · TF-IDF+SVD, loaded from S3 into memory\n"
        "- 📈 **BI Dashboard** — server-side aggregation (raw rows never leave the backend)\n"
        "- 📊 **Data Drift** — KS-test & Earth Mover's Distance between listing snapshots\n"
    ),
    version="2.1.0",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

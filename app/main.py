import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

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

# Interactive docs are a development tool, not a public surface. The frontend reaches
# this API through its own server-side proxy (Next.js app/api/[...path]), so nothing
# in production needs /docs, /redoc or /openapi.json — and serving them publishes the
# full request schema to anyone who finds the origin. Set ENV=development to get them
# back locally; on Railway ENV is unset, so all three resolve to None and 404.
_DEV = settings.ENV.lower() in ("development", "dev", "local")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "## Car Price Prediction & Analytics API\n\n"
        "Serving layer for a used-car price model trained on 29,988 deduplicated "
        "Turkish listings (BMW + Audi). Every figure the site quotes comes from the "
        "same evaluation this model shipped from.\n\n"
        "- 🚗 **Price prediction** — LightGBM · TF-IDF+SVD, bundled with its vectorizer "
        "and category maps and loaded from S3 into memory at boot. Trained on `log1p(price)`; "
        "returns a point estimate with a fixed ±6.6% band.\n"
        "- 📈 **BI dashboard** — aggregation runs server-side; raw listing rows never "
        "leave the backend.\n"
        "- 📊 **Data drift** — KS test + Wasserstein distance between listing snapshots "
        "(not between model versions).\n\n"
        "**Scope.** Premium German segment, Turkish market, Jan–Jun 2026 snapshots. "
        "The 5-fold out-of-fold MAPE is 6.49%; the interval is a fixed band, so it "
        "under-covers the cheapest quartile (81.7% against a 90% target). "
        "There is no model registry and no quantile head — one model, one bundle."
    ),
    version="2.1.0",
    openapi_tags=tags_metadata,
    docs_url="/docs" if _DEV else None,
    redoc_url="/redoc" if _DEV else None,
    openapi_url="/openapi.json" if _DEV else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Tell search engines not to index any endpoint (this is a private API surface for
# the frontend, not content). Stamped on every response — docs, openapi, /api/*.
@app.middleware("http")
async def add_noindex_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
    return response


@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    """Disallow all crawlers from the whole API."""
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

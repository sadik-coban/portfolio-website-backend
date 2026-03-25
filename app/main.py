import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router
from app.services.predict_service import preload_latest_model, unload_models

os.makedirs(settings.STATIC_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application... Beginning model preload from S3.")
    try:
        preload_latest_model()
        print("Latest model loaded into memory from S3 and ready.")
    except Exception as e:
        print(f"Failed to load model during startup: {e}")
        print("Model will be loaded on first request.")

    yield

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

app.mount("/reports", StaticFiles(directory=settings.STATIC_DIR), name="reports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

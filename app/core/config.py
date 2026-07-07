import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: config.py -> core -> app -> <root>
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Persistent volume root. On Railway mount a Volume at /data and set
# VOLUME_DIR=/data. Locally defaults to the repo root so existing files
# (cars.duckdb) keep working.
_VOLUME_DIR = os.getenv("VOLUME_DIR", str(_REPO_ROOT))


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Car Price Prediction & MLOps API"
    # CORS: only the portfolio frontend (prod + local dev). Env-overridable so Railway
    # can adjust without a redeploy. Explicit origins (not "*") also fix the latent
    # allow_credentials=True + "*" spec violation.
    ALLOWED_ORIGINS: list[str] = [
        "https://www.sadikcoban.com",
        "https://sadikcoban.com",
        "http://localhost:3000",
        "http://localhost:3737",
    ]

    # Persistent volume — single source of truth for the data + model artifacts.
    #   {VOLUME_DIR}/cars.duckdb   (pulled from S3 once at boot)
    VOLUME_DIR: str = _VOLUME_DIR
    DUCKDB_PATH: str = str(Path(_VOLUME_DIR) / "cars.duckdb")

    # DuckDB resource caps (applied as PRAGMAs per connection in core/db.py).
    # memory_limit spills to disk past the cap → protects a small Railway box
    # from OOM during full-scan dedups / .df() materializations.
    DUCKDB_MEMORY_LIMIT: str = "512MB"
    DUCKDB_THREADS: int = 2

    # Railway S3 (S3-compatible). Source for the serving model + cars.duckdb.
    RAILWAY_S3_ENDPOINT: str = ""
    RAILWAY_S3_ACCESS_KEY: str = ""
    RAILWAY_S3_SECRET_KEY: str = ""
    RAILWAY_S3_BUCKET: str = ""

    # One-time cars.duckdb bootstrap key (S3 → volume at startup).
    DATA_S3_KEY: str = "data/cars.duckdb"

    # Price-prediction serving model (best model = LightGBM · TF-IDF+SVD). The pickle
    # bundles the LightGBM model + its preprocessing (tfidf/svd + cat_maps + feat_cols).
    # NOT committed to git — downloaded from S3 on startup and held in memory.
    SERVING_MODEL_KEY: str = "serving/lightgbm_tfidf_svd.pkl"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

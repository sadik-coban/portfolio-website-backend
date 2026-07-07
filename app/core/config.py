import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: config.py -> core -> app -> api_v2 -> <root>
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Persistent volume root. On Railway mount a Volume at /data and set
# VOLUME_DIR=/data. Locally defaults to the repo root so existing files
# (cars.duckdb, registry.json, <version>/model.cbm) keep working.
_VOLUME_DIR = os.getenv("VOLUME_DIR", str(_REPO_ROOT))


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Car Price Prediction & MLOps API"
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Persistent volume — single source of truth for data + model artifacts.
    #   {VOLUME_DIR}/cars.duckdb
    #   {VOLUME_DIR}/registry.json
    #   {VOLUME_DIR}/<version_id>/model.cbm | metrics.json | shap_summary.png | train_data.parquet
    VOLUME_DIR: str = _VOLUME_DIR

    # Local DuckDB (listing data + pre-stored aggregates) lives on the volume.
    # Built by pipeline/build_duckdb.py + pipeline/build_aggregates.py, uploaded
    # via POST /admin/data/upload.
    DUCKDB_PATH: str = str(Path(_VOLUME_DIR) / "cars.duckdb")

    # Admin API security (see app/core/security.py)
    ADMIN_API_KEY: str = ""          # required to call /admin/*; empty => admin disabled (503)
    ADMIN_IP_ALLOWLIST: str = ""     # comma-separated IPs; empty => IP check off

    # Model RAM cache. False => never keep a CatBoost model resident; each
    # /predict loads the .cbm fresh from the volume and drops it after use
    # (minimal RAM footprint, slower per request). Flip to True to re-enable
    # the in-memory loaded_models cache + startup preload.
    MODEL_CACHE_ENABLED: bool = False

    # Railway S3 bucket — the curated serving store and model source for
    # POST /admin/models/sync-s3. Populated local→S3 by
    # pipeline/publish_model_to_s3.py (models) + publish_data_to_s3.py (data).
    # api_v2 pulls model versions S3 → volume and polls S3 for data refresh.
    RAILWAY_S3_ENDPOINT: str = ""
    RAILWAY_S3_ACCESS_KEY: str = ""
    RAILWAY_S3_SECRET_KEY: str = ""
    RAILWAY_S3_BUCKET: str = ""

    # Data refresh — api_v2 polls a small S3 manifest and pulls a new cars.duckdb
    # only when its version changes (atomic swap). 0 disables the poll loop (data
    # then only enters via the manual POST /admin/data/upload). Each poll tick is
    # just a tiny manifest GET; the heavy cars.duckdb download happens once per
    # version change. See app/services/data_sync_service.py.
    DATA_SYNC_POLL_SECONDS: int = 0  # data-sync poll deactivated for now (2026-07-07)
    DATA_S3_KEY: str = "data/cars.duckdb"
    DATA_MANIFEST_KEY: str = "data/manifest.json"

    # Price-prediction serving model (best model = LightGBM · TF-IDF+SVD). The pickle
    # bundles the LightGBM model + its preprocessing (tfidf/svd + cat_maps + feat_cols).
    # It is NOT committed to git — downloaded from S3 on startup and held in memory.
    SERVING_MODEL_KEY: str = "serving/lightgbm_tfidf_svd.pkl"

    # Internal paths (for static reports served locally)
    STATIC_DIR: str = "static_reports"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

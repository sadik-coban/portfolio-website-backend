import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Car Price Prediction & MLOps API"
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Railway S3 Bucket
    RAILWAY_S3_ENDPOINT: str = ""
    RAILWAY_S3_ACCESS_KEY: str = ""
    RAILWAY_S3_SECRET_KEY: str = ""
    RAILWAY_S3_BUCKET: str = ""

    # Internal paths (for static reports served locally)
    STATIC_DIR: str = "static_reports"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

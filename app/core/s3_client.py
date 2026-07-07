"""S3 client for the Railway bucket — read-only.

Downloads the serving model bundle from S3 into memory on startup, and (once,
at boot) cars.duckdb onto the volume.
"""
import os
import tempfile

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


def get_s3_client():
    """Create a boto3 S3 client targeting the Railway S3-compatible endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=settings.RAILWAY_S3_ENDPOINT,
        aws_access_key_id=settings.RAILWAY_S3_ACCESS_KEY,
        aws_secret_access_key=settings.RAILWAY_S3_SECRET_KEY,
    )


def download_to_tempfile(key: str, suffix: str = "") -> str:
    """Download an S3 object to a temp file. Returns the temp file path."""
    s3 = get_s3_client()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        s3.download_fileobj(settings.RAILWAY_S3_BUCKET, key, tmp)
        tmp.close()
        return tmp.name
    except ClientError as e:
        tmp.close()
        os.unlink(tmp.name)
        raise FileNotFoundError(f"S3 key not found: {key}") from e

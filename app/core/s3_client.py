"""
S3 Client for Railway Bucket — READ operations for api_v2.
Downloads parquet, model, registry, and SHAP files from S3.
"""
import os
import io
import json
import tempfile
import boto3
from botocore.exceptions import ClientError
from functools import lru_cache
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


def read_s3_bytes(key: str) -> bytes:
    """Read raw bytes from an S3 object."""
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=settings.RAILWAY_S3_BUCKET, Key=key)
        return response["Body"].read()
    except ClientError as e:
        raise FileNotFoundError(f"S3 key not found: {key}") from e


def read_s3_json(key: str):
    """Read and parse a JSON file from S3."""
    data = read_s3_bytes(key)
    return json.loads(data)


def list_objects(prefix: str = "") -> list[str]:
    """List all object keys under a given prefix."""
    s3 = get_s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.RAILWAY_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def object_exists(key: str) -> bool:
    """Check if an S3 object exists."""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=settings.RAILWAY_S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False

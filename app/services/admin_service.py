"""Admin operations on the persistent volume (model + data lifecycle).

All functions are pure volume/S3 operations; auth is enforced at the router
level (app/core/security.py). Used by app/api/admin_routes.py.

Source of truth = the Railway S3 bucket (the curated serving store). Models are
published local→S3 by pipeline/publish_model_to_s3.py and data by
pipeline/publish_data_to_s3.py. This service pulls the curated artifacts
S3 → volume so api_v2 serves them locally.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import duckdb

from app.core.config import settings
from app.core import s3_client
from app.services import predict_service

# Per-version artifacts to pull S3 → volume. publish_model_to_s3.py converts
# train/test CSV → parquet on upload, so we download the parquet directly
# (matches what drift_service expects).
_S3_VERSION_FILES = ["model.cbm", "train_data.parquet", "test_data.parquet", "metrics.json", "shap_summary.png"]
# Tables a freshly-built cars.duckdb must contain to be accepted.
_REQUIRED_TABLES = ("car_listings", "dashboard_cache", "options_cache")


def _volume() -> Path:
    p = Path(settings.VOLUME_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_registry() -> list[dict]:
    path = _volume() / "registry.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(registry: list[dict]) -> None:
    path = _volume() / "registry.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Status ───────────────────────────────────────────────────────────

def get_models_status() -> dict:
    """Registry + on-disk presence per version, plus loaded-in-memory set."""
    vol = _volume()
    registry = _read_registry()
    versions = []
    for v in sorted(registry, key=lambda x: x.get("date", ""), reverse=True):
        vid = v.get("version_id")
        vdir = vol / str(vid)
        versions.append({
            "version_id": vid,
            "date": v.get("date"),
            "metrics": v.get("metrics", {}),
            "files": {
                "model.cbm": (vdir / "model.cbm").exists(),
                "metrics.json": (vdir / "metrics.json").exists(),
                "shap_summary.png": (vdir / "shap_summary.png").exists(),
                "train_data.parquet": (vdir / "train_data.parquet").exists(),
            },
            "loaded_in_memory": vid in predict_service.loaded_models,
        })
    return {"volume_dir": str(vol), "count": len(versions), "versions": versions}


def health() -> dict:
    vol = _volume()
    duck = Path(settings.DUCKDB_PATH)
    return {
        "volume_dir": str(vol),
        "volume_exists": vol.exists(),
        "duckdb_path": str(duck),
        "duckdb_exists": duck.exists(),
        "duckdb_size": duck.stat().st_size if duck.exists() else None,
        "registry_versions": [v.get("version_id") for v in _read_registry()],
        "models_loaded": sorted(predict_service.loaded_models.keys()),
        "admin_configured": bool(settings.ADMIN_API_KEY),
    }


# ── Model add (S3 → volume) ──────────────────────────────────────────

def sync_s3(version_ids: list[str] | None = None, sync_registry: bool = True) -> dict:
    """Download versions from the S3 bucket onto the volume + merge registry.

    S3 is the curated serving store (fed by pipeline/publish_model_to_s3.py).
    version_ids None/empty → sync ALL versions in the S3 registry
    (keep every served version local on the volume).
    """
    if not settings.RAILWAY_S3_BUCKET:
        raise RuntimeError("RAILWAY_S3_BUCKET is not configured.")

    vol = _volume()
    errors: list[str] = []

    # S3 registry entries (source of truth for the version metadata we merge in).
    s3_registry: list[dict] = []
    try:
        s3_registry = s3_client.read_s3_json("registry.json")
    except Exception as e:
        errors.append(f"registry.json (S3): {e}")

    s3_by_id = {v.get("version_id"): v for v in s3_registry}

    # Default: all versions from the S3 registry.
    if not version_ids:
        version_ids = [v.get("version_id") for v in s3_registry if v.get("version_id")]
        if not version_ids:
            raise RuntimeError("S3 registry empty or unreadable; no versions to sync.")

    for vid in version_ids:
        vdir = vol / vid
        vdir.mkdir(parents=True, exist_ok=True)
        for filename in _S3_VERSION_FILES:
            key = f"{vid}/{filename}"
            try:
                tmp = s3_client.download_to_tempfile(key)
                # Move into place on the volume (handles cross-filesystem temp).
                shutil.move(tmp, str(vdir / filename))
            except FileNotFoundError:
                errors.append(f"{key}: not found in S3")
            except Exception as e:
                errors.append(f"{key}: {e}")

    # Merge registry entries (insert/update by version_id), like upload_to_server.py.
    if sync_registry:
        registry = _read_registry()
        by_id = {v.get("version_id"): i for i, v in enumerate(registry)}
        for vid in version_ids:
            entry = s3_by_id.get(vid, {"version_id": vid})
            if vid in by_id:
                registry[by_id[vid]] = entry
            else:
                registry.insert(0, entry)
        _write_registry(registry)

    # Refresh in-memory copies of any re-synced model already loaded.
    for vid in version_ids:
        if vid in predict_service.loaded_models:
            predict_service.unload_model(vid)

    return {"synced": version_ids, "errors": errors}


# ── Model delete ─────────────────────────────────────────────────────

def delete_model(version_id: str) -> dict:
    """Remove a version dir + registry entry + evict from memory. Guards last version."""
    registry = _read_registry()
    ids = [v.get("version_id") for v in registry]
    if version_id not in ids and not (_volume() / version_id).exists():
        raise FileNotFoundError(f"Version '{version_id}' not found.")
    if len(ids) <= 1 and version_id in ids:
        raise ValueError("Refusing to delete the only remaining model version.")

    vdir = _volume() / version_id
    if vdir.exists():
        shutil.rmtree(vdir)
    registry = [v for v in registry if v.get("version_id") != version_id]
    _write_registry(registry)
    predict_service.unload_model(version_id)
    return {"deleted": version_id, "remaining": [v.get("version_id") for v in registry]}


# ── Data update (upload prebuilt cars.duckdb → atomic swap) ──────────

def _validate_duckdb(path: Path) -> dict:
    """Open read-only and assert required tables + non-empty car_listings.

    Raises ValueError for any invalid/corrupt file or missing content so the
    route can map it to HTTP 400 (DuckDB itself raises IOException/etc.).
    """
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as e:
        raise ValueError(f"not a valid DuckDB file ({e})")
    try:
        names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = [t for t in _REQUIRED_TABLES if t not in names]
        if missing:
            raise ValueError(f"missing required tables: {', '.join(missing)}")
        rows = con.execute("SELECT count(*) FROM car_listings").fetchone()[0]
        if not rows:
            raise ValueError("car_listings is empty")
        return {"tables": sorted(names), "car_listings_rows": rows}
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"validation query failed ({e})")
    finally:
        con.close()


def _atomic_install_duckdb(src_path) -> dict:
    """Validate a cars.duckdb already staged on the volume filesystem and
    atomically swap it into DUCKDB_PATH.

    Shared by upload_data (multipart) and data_sync_service (S3 poll). `src_path`
    MUST live on the same filesystem as DUCKDB_PATH so `os.replace` is atomic.
    On validation failure the live DB is left untouched; the caller owns
    `src_path` cleanup.
    """
    src = Path(src_path)
    dest = Path(settings.DUCKDB_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = _validate_duckdb(src)         # raises ValueError on invalid/corrupt
    os.replace(src, dest)                # atomic swap (same filesystem)
    return {"duckdb_path": str(dest), **info}


def upload_data(upload_file) -> dict:
    """Stream an uploaded cars.duckdb to the volume, validate, then atomic-swap.

    `upload_file` is a Starlette UploadFile. On any validation failure the live
    DB is left untouched.
    """
    dest = Path(settings.DUCKDB_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Temp on the SAME filesystem so os.replace() is atomic.
    fd, tmp_name = tempfile.mkstemp(suffix=".duckdb.tmp", dir=str(dest.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(upload_file.file, out)
        return _atomic_install_duckdb(tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

"""Data refresh — poll a small S3 manifest and pull a new cars.duckdb onto the
volume (atomic swap) only when its version changes.

No inbound endpoint: api_v2 reads S3 itself (with the read-only creds it already
has), so there is nothing to spam or to steal. Each poll tick is just a tiny
`manifest.json` GET; the heavy cars.duckdb download happens once per version
change. The applied version is persisted on the volume so restarts/new instances
are idempotent.

Wired into app.main lifespan when settings.DATA_SYNC_POLL_SECONDS > 0.
"""
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core import s3_client
from app.services import admin_service

_APPLIED_VERSION_FILE = ".data_version.json"
# Single-flight: never run two syncs concurrently (poll tick vs bootstrap).
_sync_lock = asyncio.Lock()


def _volume() -> Path:
    return Path(settings.VOLUME_DIR)


def _applied_version_path() -> Path:
    return _volume() / _APPLIED_VERSION_FILE


def current_applied_version():
    """The data version currently installed on the volume (None if unknown)."""
    p = _applied_version_path()
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("version")
    except Exception:
        return None


def _write_applied_version(version, info: dict | None = None) -> None:
    p = _applied_version_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    payload = {"version": version}
    if info:
        payload["installed"] = info
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_to_volume_temp(key: str) -> Path:
    """Stream an S3 object to a temp file on the volume filesystem so the later
    `os.replace` into DUCKDB_PATH is atomic (same filesystem)."""
    dest_dir = Path(settings.DUCKDB_PATH).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".duckdb.dl", dir=str(dest_dir))
    tmp = Path(tmp_name)
    try:
        s3 = s3_client.get_s3_client()
        with os.fdopen(fd, "wb") as out:
            s3.download_fileobj(settings.RAILWAY_S3_BUCKET, key, out)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return tmp


def _sync_data_from_s3_locked(force: bool = False) -> dict:
    """Blocking core of the sync (run via asyncio.to_thread under _sync_lock).

    Reads the manifest, and if its version differs from the applied version,
    downloads cars.duckdb, verifies sha256 (if present), validates the tables,
    and atomically swaps it into place. On any failure the live DB is untouched.
    """
    if not settings.RAILWAY_S3_BUCKET:
        raise RuntimeError("RAILWAY_S3_BUCKET is not configured.")

    manifest = s3_client.read_s3_json(settings.DATA_MANIFEST_KEY)
    target_version = manifest.get("version")
    if target_version is None:
        raise ValueError(f"manifest {settings.DATA_MANIFEST_KEY} has no 'version'")

    applied = current_applied_version()
    if not force and applied is not None and str(applied) == str(target_version):
        return {"updated": False, "version": applied, "reason": "up-to-date"}

    tmp = _download_to_volume_temp(settings.DATA_S3_KEY)
    try:
        expected_sha = manifest.get("sha256")
        if expected_sha:
            actual = _sha256(tmp)
            if actual.lower() != str(expected_sha).lower():
                raise ValueError(f"sha256 mismatch (expected {expected_sha}, got {actual})")
        # Reuse: validate required tables + atomic os.replace swap.
        info = admin_service._atomic_install_duckdb(tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    _write_applied_version(target_version, info)
    return {"updated": True, "from": applied, "to": target_version, **info}


async def sync_data_from_s3(force: bool = False) -> dict:
    """Pull cars.duckdb S3 → volume if the manifest version changed.

    Single-flight (asyncio lock); the blocking work runs in a worker thread so
    the event loop is never blocked.
    """
    async with _sync_lock:
        return await asyncio.to_thread(_sync_data_from_s3_locked, force)


async def poll_loop() -> None:
    """Background task: bootstrap once, then poll the manifest every N seconds."""
    interval = settings.DATA_SYNC_POLL_SECONDS
    try:
        res = await sync_data_from_s3()
        print(f"[data-sync] bootstrap: {res}")
    except Exception as e:
        print(f"[data-sync] bootstrap failed (serving existing DB if any): {e}")

    while True:
        await asyncio.sleep(interval)
        try:
            res = await sync_data_from_s3()
            if res.get("updated"):
                print(f"[data-sync] updated: {res}")
        except Exception as e:
            print(f"[data-sync] poll error: {e}")

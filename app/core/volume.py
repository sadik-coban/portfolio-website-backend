"""Start every deploy from an empty volume.

The data is frozen, so nothing on the volume is worth keeping across a deploy: cars.duckdb is
pulled from S3 again at boot and the drift cache is rebuilt from it (drift_service.warm_cache).
Keeping them is what let a stale database or a cache written by older code outlive the deploy
that should have replaced it.

A *deploy* is told apart from a *restart* (restartPolicy ON_FAILURE) by Railway's
RAILWAY_DEPLOYMENT_ID, which changes on every deploy: the id is written to the volume after a
wipe, and a boot that finds the same id there leaves the volume alone.

Guards — this deletes files, so it only ever runs on a real Railway volume:
  • RAILWAY_DEPLOYMENT_ID must be set (it never is locally);
  • the volume must be named explicitly — VOLUME_DIR, or the RAILWAY_VOLUME_MOUNT_PATH Railway
    injects when a volume is attached (/data). Without either, VOLUME_DIR falls back to the repo
    root, which locally is the whole portfolio checkout and on Railway is the container's "/";
  • and the resolved volume must not be the repo root anyway.
"""
import os
import shutil
from pathlib import Path

from app.core.config import settings

_MARKER = ".deployment_id"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_KEEP = {"lost+found"}  # filesystem bookkeeping on an ext4 mount, not ours


def reset_on_new_deploy() -> None:
    deployment = os.getenv("RAILWAY_DEPLOYMENT_ID")
    if not deployment:
        return  # not on Railway (local run): never touch anything
    if not (os.getenv("VOLUME_DIR") or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")):
        print("Volume reset skipped: no volume attached (neither VOLUME_DIR nor RAILWAY_VOLUME_MOUNT_PATH is set).")
        return
    volume = Path(settings.VOLUME_DIR).resolve()
    if volume == _REPO_ROOT or not volume.is_dir():
        print(f"Volume reset skipped: {volume} is the repo root or not a directory.")
        return

    marker = volume / _MARKER
    try:
        if marker.read_text(encoding="utf-8").strip() == deployment:
            print(f"Same deployment ({deployment}) — a restart, volume kept.")
            return
    except FileNotFoundError:
        pass

    print(f"New deployment ({deployment}) — emptying the volume {volume}.")
    for entry in volume.iterdir():
        if entry.name in _KEEP:
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            print(f"  removed {entry.name}")
        except Exception as e:  # keep booting: a leftover file is better than no service
            print(f"  could not remove {entry.name} ({e})")
    marker.write_text(deployment, encoding="utf-8")

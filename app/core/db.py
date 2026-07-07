"""Local DuckDB access for the listing data.

The data source is a local DuckDB file (settings.DUCKDB_PATH), pulled from S3
onto the volume once at startup. Read paths open the file read-only, so multiple
requests can query concurrently and the file is never mutated by the API.
"""
import os
from contextlib import contextmanager

import duckdb

from app.core.config import settings


def db_exists() -> bool:
    return os.path.exists(settings.DUCKDB_PATH)


@contextmanager
def get_db_connection(read_only: bool = True):
    """Yield a read-only DuckDB connection to the local cars.duckdb file.

    Raises FileNotFoundError (→ 404/500 at the route) if the DB has not been
    provisioned yet, instead of silently creating an empty database. Caps
    DuckDB's memory (spills to disk past the limit) and thread count so a big
    scan can't OOM / oversubscribe a small Railway instance.
    """
    if read_only and not db_exists():
        raise FileNotFoundError(
            f"DuckDB not found at {settings.DUCKDB_PATH}. "
            "It is pulled from S3 onto the volume at startup."
        )
    conn = duckdb.connect(settings.DUCKDB_PATH, read_only=read_only)
    try:
        # Best-effort resource caps (older DuckDB builds may reject a PRAGMA).
        try:
            conn.execute(f"PRAGMA memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
            conn.execute(f"PRAGMA threads={settings.DUCKDB_THREADS}")
        except Exception as e:
            print(f"DuckDB PRAGMA skipped: {e}")
        yield conn
    finally:
        conn.close()

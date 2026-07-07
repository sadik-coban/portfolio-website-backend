"""Local DuckDB access for the listing data + pre-stored aggregates.

The data source is a local DuckDB file (settings.DUCKDB_PATH) produced by:
    1. pipeline/build_duckdb.py     → car_listings table (audi + bmw, all rows)
    2. pipeline/build_aggregates.py → dashboard_cache, options_cache tables

Read paths open the file read-only, so multiple requests can query
concurrently and the file is never mutated by the API.
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
    built yet, instead of silently creating an empty database.
    """
    if read_only and not db_exists():
        raise FileNotFoundError(
            f"DuckDB not found at {settings.DUCKDB_PATH}. "
            "Run pipeline/build_duckdb.py and pipeline/build_aggregates.py first."
        )
    conn = duckdb.connect(settings.DUCKDB_PATH, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return row is not None

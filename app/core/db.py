import duckdb
from contextlib import contextmanager

# This module manages the DuckDB connection. 
# DuckDB can be used concurrently, but depending on the exact workloads 
# connections should be managed efficiently.

# An in-memory duckdb connection is often enough for reading external parquet files.
# It can be persistent if we need it to be, but DuckDB allows reading parquet files directly.

@contextmanager
def get_db_connection():
    """
    Yields a DuckDB connection. 
    Using an in-memory database configuration since our primary data source 
    are parquet files stored on the volume.
    """
    conn = duckdb.connect(':memory:')
    try:
        yield conn
    finally:
        conn.close()

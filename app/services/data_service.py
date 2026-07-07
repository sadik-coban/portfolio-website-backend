"""Snapshot dates + the shared time-source relation, served from local DuckDB.

Only two things survive from the old dashboard service now that the frontend
reads the dashboard from /api/bi/*:
  • get_snapshots() — the snapshot-date list (with per-snapshot + cumulative
    counts) for the drift page's time selector. Cached in-process: the result
    only changes when cars.duckdb is swapped, which now happens once at boot.
  • _time_source() / _UNTIL_LATEST — the point-in-time relation that
    drift_service.get_data_drift substitutes for car_listings to de-duplicate
    to one row per ad_id.
"""
from app.core.db import get_db_connection


# ── Time dimension (snapshot selection) ──────────────────────────────
# A chart/drift query runs against a *time-source* relation instead of raw
# car_listings:
#   mode="specific" → only rows scraped on `as_of` (search_date = as_of)
#   mode="until"    → point-in-time: for each ad_id, its latest version with
#                     search_date <= as_of (or over all snapshots when as_of is None).
_UNTIL_LATEST = (
    "SELECT * EXCLUDE(rn) FROM ("
    "SELECT *, row_number() OVER (PARTITION BY ad_id ORDER BY search_date DESC, scraped_at DESC) AS rn "
    "FROM car_listings{where}) WHERE rn = 1"
)


def _time_source(as_of=None, mode: str = "until") -> tuple[str, dict]:
    """Return (relation_sql, params) to substitute for `car_listings`.

    Default (as_of None) = point-in-time current state across all snapshots
    (one row per ad_id), which de-duplicates the listing data.
    """
    if mode == "specific" and as_of:
        return "(SELECT * FROM car_listings WHERE search_date = $as_of)", {"as_of": as_of}
    if as_of:  # until a given date
        return "(" + _UNTIL_LATEST.format(where=" WHERE search_date <= $as_of") + ")", {"as_of": as_of}
    # until over everything (current state)
    return "(" + _UNTIL_LATEST.format(where="") + ")", {}


# ── Snapshot dates (cached — the DB only changes on the boot bootstrap) ──
_SNAPSHOTS = None


def get_snapshots(force: bool = False):
    """Available snapshot dates with specific (per-snapshot) + until (cumulative)
    counts. Cached in-process: the answer only changes when cars.duckdb is
    swapped, which now happens once at startup — so the `1 + 2N` COUNT queries
    run at most once per process instead of on every drift-page open."""
    global _SNAPSHOTS
    if _SNAPSHOTS is not None and not force:
        return _SNAPSHOTS
    try:
        with get_db_connection() as conn:
            dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT search_date FROM car_listings ORDER BY 1"
            ).fetchall()]
            out = []
            for d in dates:
                specific = conn.execute(
                    "SELECT count(DISTINCT ad_id) FROM car_listings WHERE search_date = ?", [d]
                ).fetchone()[0]
                until = conn.execute(
                    "SELECT count(DISTINCT ad_id) FROM car_listings WHERE search_date <= ?", [d]
                ).fetchone()[0]
                out.append({
                    "date": str(d),
                    "specific_count": specific,
                    "until_count": until,
                })
            _SNAPSHOTS = {"snapshots": out, "latest": str(dates[-1]) if dates else None}
            return _SNAPSHOTS
    except FileNotFoundError as e:
        return {"error": str(e), "snapshots": []}  # not cached — retry next call

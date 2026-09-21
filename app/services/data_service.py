"""Snapshot dates + the shared time-source relation, served from local DuckDB.

Only two things survive from the old dashboard service now that the frontend
reads the dashboard from /api/bi/*:
  • get_snapshots() — the snapshot-date list (with per-snapshot + cumulative
    counts) for the drift page's time selector. Cached in-process: the result
    only changes when cars.duckdb is swapped, which now happens once at boot.
  • _time_source() / _UNTIL_LATEST — the point-in-time relation that
    drift_service.get_data_drift substitutes for car_listings to de-duplicate
    to one row per ad_id.
  • TR_ROWS — the predicate that decides which listings count at all.
"""
from app.core.db import get_db_connection


# ── The canonical row set ────────────────────────────────────────────
# Every surface that reports a number about the listing population — the BI
# dashboard, the drift page, and the offline pipeline that writes the technical
# report — has to agree on which rows count. This predicate is that agreement.
#
# gb_plate_origin reads like a market segmentation but is not one: it holds
# '(TR) Türkiye' on 45,159 rows and NULL on 118, with no foreign-plate value
# anywhere in the table. Those 118 are scrapes where a whole block of the listing
# page failed to parse — gb_usage_type is NULL on exactly the same rows and on no
# others, and 23 of the 77 ads involved appear fully parsed on another date. They
# skew cheap (median ₺1.1M against ₺1.52M overall), so leaving them in visibly
# moves a price distribution: the drift page answered 48,087.68 where the report
# said ₺48,059, and this one missing predicate was the whole of that gap.
#
# bi_service builds its dashboard view from this same constant. Keep it that way —
# the bug above was a second copy of this string that someone forgot to write.
TR_ROWS = "price > 0 AND gb_plate_origin = '(TR) Türkiye'"


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

    TR_ROWS goes inside the subquery, before the de-duplication window — the
    order bi_service uses. Filtering after the window would keep an ad whose
    latest scrape lost the block and throw away its earlier, intact rows.
    """
    if mode == "specific" and as_of:
        return f"(SELECT * FROM car_listings WHERE search_date = $as_of AND {TR_ROWS})", {"as_of": as_of}
    if as_of:  # until a given date
        return "(" + _UNTIL_LATEST.format(where=f" WHERE search_date <= $as_of AND {TR_ROWS}") + ")", {"as_of": as_of}
    # until over everything (current state)
    return "(" + _UNTIL_LATEST.format(where=f" WHERE {TR_ROWS}") + ")", {}


# ── Snapshot dates (cached — the DB only changes on the boot bootstrap) ──
_SNAPSHOTS = None


def get_snapshots(force: bool = False):
    """Available snapshot dates with specific (per-snapshot) + until (cumulative)
    counts. Cached in-process: the answer only changes when cars.duckdb is
    swapped, which now happens once at startup — so the `1 + 2N` COUNT queries
    run at most once per process instead of on every drift-page open.

    Counted over TR_ROWS, the same rows _time_source hands to the drift stats —
    otherwise the date picker advertises a sample size the numbers beside it were
    not computed from."""
    global _SNAPSHOTS
    if _SNAPSHOTS is not None and not force:
        return _SNAPSHOTS
    try:
        with get_db_connection() as conn:
            dates = [r[0] for r in conn.execute(
                f"SELECT DISTINCT search_date FROM car_listings WHERE {TR_ROWS} ORDER BY 1"
            ).fetchall()]
            out = []
            for d in dates:
                specific = conn.execute(
                    f"SELECT count(DISTINCT ad_id) FROM car_listings WHERE search_date = ? AND {TR_ROWS}", [d]
                ).fetchone()[0]
                until = conn.execute(
                    f"SELECT count(DISTINCT ad_id) FROM car_listings WHERE search_date <= ? AND {TR_ROWS}", [d]
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

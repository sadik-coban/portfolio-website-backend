"""Dashboard & dropdown data — served from the local DuckDB file.

Two layers:
  • Pre-stored aggregates (fast path): pipeline/build_aggregates.py materializes
    the full dashboard payload for every (brand, series) scope into
    `dashboard_cache`, and dropdown options into `options_cache`. When a request
    carries no range filters (only brand/series), the payload is pulled directly
    from those tables — no scan of the 33k-row listing table.
  • Live aggregation (fallback): when range filters (price/year/km/fuel) are
    present, `_compute_dashboard` runs the aggregation against `car_listings`.

The chart shapes are kept identical to the previous S3/parquet service so the
frontend contract is unchanged; only the source (local DuckDB) and the column
names (real silver schema) differ.
"""
import json
from typing import Optional

from app.core.db import get_db_connection, table_exists

ALL = "__ALL__"

# Damage part label → column prefix in car_listings. ALL 13 parts (bumpers incl.).
# Each part carries its full breakdown: degisen / boyali / lokal (+ total value).
DAMAGE_PARTS = {
    "Kaput": "kaput",
    "Tavan": "tavan",
    "Bagaj": "bagaj",
    "Ön Tampon": "bumper_front",
    "Arka Tampon": "bumper_rear",
    "Sol Ön Kapı": "door_fl",
    "Sağ Ön Kapı": "door_fr",
    "Sol Arka Kapı": "door_rl",
    "Sağ Arka Kapı": "door_rr",
    "Sol Ön Çamurluk": "fender_fl",
    "Sağ Ön Çamurluk": "fender_fr",
    "Sol Arka Çamurluk": "fender_rl",
    "Sağ Arka Çamurluk": "fender_rr",
}


# ── Filtering ────────────────────────────────────────────────────────

def build_filter_clause(
    brand, series, min_price, max_price, min_year, max_year,
    min_km, max_km, fuel, include_series=True,
):
    filters = ["1=1"]
    params = {}
    if brand and brand != "Tümü":
        filters.append("brand = $brand")
        params["brand"] = brand
    if include_series and series and series != "Tümü":
        filters.append("series = $series")
        params["series"] = series
    if min_price:
        filters.append("price >= $min_price")
        params["min_price"] = min_price
    if max_price:
        filters.append("price <= $max_price")
        params["max_price"] = max_price
    if min_year:
        filters.append("kb_year >= $min_year")
        params["min_year"] = min_year
    if max_year:
        filters.append("kb_year <= $max_year")
        params["max_year"] = max_year
    if min_km:
        filters.append("kb_mileage >= $min_km")
        params["min_km"] = min_km
    if max_km:
        filters.append("kb_mileage <= $max_km")
        params["max_km"] = max_km
    if fuel and fuel != "Tümü":
        filters.append("kb_fuel = $fuel")
        params["fuel"] = fuel
    return " AND ".join(filters), params


def _only_brand_series(min_price, max_price, min_year, max_year, min_km, max_km, fuel) -> bool:
    """True when the request can be served from the pre-stored cache."""
    return not any([min_price, max_price, min_year, max_year, min_km, max_km, fuel])


# ── Time dimension (snapshot selection) ──────────────────────────────
# Every chart query runs against a *time-source* relation instead of the raw
# car_listings table:
#   mode="specific" → only the rows scraped on `as_of` (search_date = as_of)
#   mode="until"    → point-in-time: for each ad_id, its latest version with
#                     search_date <= as_of (or over all snapshots when as_of is None).
MODES = ("specific", "until")
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


# ── Core aggregation (shared by build script and live fallback) ──────

def compute_dashboard(
    conn,
    brand: Optional[str] = None, series: Optional[str] = None,
    min_price: Optional[float] = None, max_price: Optional[float] = None,
    min_year: Optional[int] = None, max_year: Optional[int] = None,
    min_km: Optional[int] = None, max_km: Optional[int] = None,
    fuel: Optional[str] = None,
    as_of=None, mode: str = "until",
) -> dict:
    """Build the full dashboard payload against a time-scoped view of car_listings."""
    src, t_params = _time_source(as_of, mode)

    base_where, base_params = build_filter_clause(
        brand, None, min_price, max_price, min_year, max_year,
        min_km, max_km, fuel, include_series=False,
    )
    final_where, final_params = build_filter_clause(
        brand, series, min_price, max_price, min_year, max_year,
        min_km, max_km, fuel, include_series=True,
    )
    base_params = {**base_params, **t_params}
    final_params = {**final_params, **t_params}

    # Brand/series option lists for the current (brand-only) scope.
    lists_row = conn.execute(
        f"""
        SELECT
            list(DISTINCT brand)  FILTER (WHERE brand IS NOT NULL)  AS brands,
            list(DISTINCT series) FILTER (WHERE series IS NOT NULL) AS series_list
        FROM {src} AS s WHERE {base_where}
        """,
        base_params,
    ).fetchone()
    unique_brands = sorted(lists_row[0]) if lists_row and lists_row[0] else []
    unique_series = sorted(lists_row[1]) if lists_row and lists_row[1] else []

    # Inline filtered subquery (DuckDB can't bind params inside CREATE VIEW).
    # final_params is passed to every query; all its keys appear in `final_where`/src.
    fd = f"(SELECT * FROM {src} AS s WHERE {final_where}) AS filtered_data"

    kpi = conn.execute(
        f"SELECT COUNT(*), COALESCE(AVG(price), 0) FROM {fd}", final_params
    ).fetchone()
    if not kpi or kpi[0] == 0:
        return empty_dashboard_response(unique_brands, unique_series)

    # Boxplot: {brand: [prices]}
    boxplot_data = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT brand, list(price) FROM {fd} "
            "WHERE price IS NOT NULL GROUP BY brand", final_params
        ).fetchall()
    }

    # Scatter: {brand: [[km, price], ...]}
    scatter_data = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT brand, list([kb_mileage, price]) FROM {fd} "
            "WHERE kb_mileage IS NOT NULL AND price IS NOT NULL GROUP BY brand", final_params
        ).fetchall()
    }

    # Line: avg price by year
    line_rows = conn.execute(
        f"SELECT kb_year, AVG(price) FROM {fd} "
        "WHERE kb_year IS NOT NULL GROUP BY kb_year ORDER BY kb_year", final_params
    ).fetchall()
    line_chart_data = {
        "years": [r[0] for r in line_rows],
        "prices": [float(r[1]) for r in line_rows],
    }

    # Donut: fuel distribution
    donut_data = [
        {"name": r[0], "value": r[1]}
        for r in conn.execute(
            f"SELECT kb_fuel, COUNT(*) FROM {fd} "
            "WHERE kb_fuel IS NOT NULL GROUP BY kb_fuel ORDER BY 2 DESC", final_params
        ).fetchall()
    ]

    # Damage: full per-part breakdown — degisen / boyali / lokal + total.
    # 13 parts (bumpers included), 3 sums each, in DAMAGE_PARTS order.
    damage_select = ", ".join(
        f"SUM(COALESCE({p}_degisen,0)), SUM(COALESCE({p}_boyali,0)), SUM(COALESCE({p}_lokal,0))"
        for p in DAMAGE_PARTS.values()
    )
    damage_row = conn.execute(
        f"SELECT {damage_select} FROM {fd}", final_params
    ).fetchone()
    damage_data = []
    for i, part in enumerate(DAMAGE_PARTS.keys()):
        deg = int(damage_row[i * 3] or 0)
        boy = int(damage_row[i * 3 + 1] or 0)
        lok = int(damage_row[i * 3 + 2] or 0)
        damage_data.append({
            "part": part,
            "degisen": deg,
            "boyali": boy,
            "lokal": lok,
            "value": deg + boy + lok,
        })

    # Radar: brand comparison (only when <= 5 brands in scope)
    brand_count = conn.execute(
        f"SELECT COUNT(DISTINCT brand) FROM {fd}", final_params
    ).fetchone()[0]
    if brand_count <= 5:
        gmax = conn.execute(
            "SELECT MAX(price), MAX(kb_mileage), MAX(power_hp_val), MAX(kb_fuel_cons_avg) "
            f"FROM {fd}", final_params
        ).fetchone()
        radar_rows = conn.execute(
            "SELECT brand, AVG(price), AVG(kb_mileage), AVG(power_hp_val), AVG(kb_fuel_cons_avg) "
            f"FROM {fd} GROUP BY brand", final_params
        ).fetchall()
        radar_data = {
            "indicators": [
                {"name": "Fiyat", "max": float(gmax[0] or 0) * 1.1},
                {"name": "KM", "max": float(gmax[1] or 0) * 1.1},
                {"name": "Beygir Gücü", "max": float(gmax[2] or 0) * 1.1},
                {"name": "Yakıt Tüketimi", "max": float(gmax[3] or 0) * 1.1},
            ],
            "series": [
                {"name": r[0], "value": [
                    float(r[1] or 0), float(r[2] or 0),
                    float(r[3] or 0), float(r[4] or 0),
                ]}
                for r in radar_rows
            ],
        }
    else:
        radar_data = {"indicators": [], "series": []}

    return {
        "brands": unique_brands,
        "seriesList": unique_series,
        "kpi": {"total": kpi[0] or 0, "avgPrice": float(kpi[1]) if kpi[1] else 0},
        "boxplotData": boxplot_data,
        "scatterData": scatter_data,
        "lineChartData": line_chart_data,
        "donutChartData": donut_data,
        "damageChartData": damage_data,
        "radarChartData": radar_data,
    }


def empty_dashboard_response(brands, series):
    return {
        "brands": brands,
        "seriesList": series,
        "kpi": {"total": 0, "avgPrice": 0},
        "boxplotData": {},
        "scatterData": {},
        "lineChartData": {},
        "donutChartData": [],
        "radarChartData": {},
        "damageChartData": [],
    }


# ── Public API (cache-first) ─────────────────────────────────────────

def _cache_lookup(conn, mode, as_of, brand, series):
    """Pre-stored dashboard payload for (mode, as_of, scope). None on miss."""
    b = brand if (brand and brand != "Tümü") else ALL
    s = series if (series and series != "Tümü") else ALL
    if as_of is None:
        row = conn.execute(
            "SELECT payload FROM dashboard_cache "
            "WHERE mode = ? AND as_of IS NULL AND scope_brand = ? AND scope_series = ?",
            [mode, b, s],
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT payload FROM dashboard_cache "
            "WHERE mode = ? AND as_of = ? AND scope_brand = ? AND scope_series = ?",
            [mode, as_of, b, s],
        ).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def get_dashboard_data(
    brand: Optional[str] = None, series: Optional[str] = None,
    min_price: Optional[float] = None, max_price: Optional[float] = None,
    min_year: Optional[int] = None, max_year: Optional[int] = None,
    min_km: Optional[int] = None, max_km: Optional[int] = None,
    fuel: Optional[str] = None,
    as_of=None, mode: str = "until",
):
    if mode not in MODES:
        mode = "until"
    if mode == "specific" and as_of is None:
        mode = "until"  # specific needs a date; fall back to current
    try:
        with get_db_connection() as conn:
            # Fast path: no range filters → pull pre-stored payload directly.
            if _only_brand_series(min_price, max_price, min_year, max_year, min_km, max_km, fuel) \
                    and table_exists(conn, "dashboard_cache"):
                try:
                    cached = _cache_lookup(conn, mode, as_of, brand, series)
                    if cached is not None:
                        return cached
                except Exception as e:
                    print("dashboard_cache lookup miss:", e)  # fall through to live

            # Fallback: aggregate live.
            return compute_dashboard(
                conn, brand, series, min_price, max_price,
                min_year, max_year, min_km, max_km, fuel, as_of=as_of, mode=mode,
            )
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        print("Dashboard SQL Error:", str(e))
        return empty_dashboard_response([], [])


def get_snapshots():
    """Available snapshot dates with specific (per-snapshot) + until (cumulative) counts."""
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
            return {"snapshots": out, "latest": str(dates[-1]) if dates else None}
    except FileNotFoundError as e:
        return {"error": str(e), "snapshots": []}


def get_price_history(ad_id: int):
    """Per-snapshot price/mileage history for one ad (chronological, with deltas)."""
    try:
        with get_db_connection() as conn:
            if not table_exists(conn, "price_history"):
                return {"ad_id": ad_id, "history": []}
            rows = conn.execute(
                "SELECT search_date, price, kb_mileage, listing_date, price_delta, "
                "snapshot_idx, snapshot_count FROM price_history "
                "WHERE ad_id = ? ORDER BY search_date, scraped_at", [ad_id]
            ).fetchall()
            history = [{
                "search_date": str(r[0]), "price": r[1], "kb_mileage": r[2],
                "listing_date": str(r[3]) if r[3] else None,
                "price_delta": r[4], "snapshot_idx": r[5], "snapshot_count": r[6],
            } for r in rows]
            return {"ad_id": ad_id, "history": history}
    except FileNotFoundError as e:
        return {"error": str(e), "ad_id": ad_id, "history": []}


def compute_options(
    conn, brand: Optional[str] = None, series: Optional[str] = None,
    as_of=None, mode: str = "until",
) -> dict:
    """Cascading dropdown options against a time-scoped view of car_listings.

    Runs against `_time_source(as_of, mode)` so a selected snapshot/date only
    surfaces the brands/series/models present in that time-slice (no stale,
    "alakasız" entries from older snapshots).
    """
    src, t_params = _time_source(as_of, mode)
    response = {"brands": [], "series": [], "models": []}
    response["brands"] = [
        r[0] for r in conn.execute(
            f"SELECT DISTINCT brand FROM {src} AS s WHERE brand IS NOT NULL ORDER BY brand",
            dict(t_params),
        ).fetchall()
    ]
    if brand:
        response["series"] = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT series FROM {src} AS s "
                "WHERE brand = $brand AND series IS NOT NULL ORDER BY series",
                {**t_params, "brand": brand},
            ).fetchall()
        ]
    if brand and series:
        response["models"] = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT model FROM {src} AS s "
                "WHERE brand = $brand AND series = $series AND model IS NOT NULL ORDER BY model",
                {**t_params, "brand": brand, "series": series},
            ).fetchall()
        ]
    return response


def _options_cache_lookup(conn, mode, as_of, brand, series):
    """Pre-stored dropdown options for (mode, as_of, scope). None on miss."""
    b = brand if brand else ALL
    s = series if series else ALL
    if as_of is None:
        row = conn.execute(
            "SELECT payload FROM options_cache "
            "WHERE mode = ? AND as_of IS NULL AND scope_brand = ? AND scope_series = ?",
            [mode, b, s],
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT payload FROM options_cache "
            "WHERE mode = ? AND as_of = ? AND scope_brand = ? AND scope_series = ?",
            [mode, as_of, b, s],
        ).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def get_dropdown_options(
    brand: Optional[str] = None, series: Optional[str] = None,
    as_of=None, mode: str = "until",
):
    if mode not in MODES:
        mode = "until"
    if mode == "specific" and as_of is None:
        mode = "until"  # specific needs a date; fall back to current
    try:
        with get_db_connection() as conn:
            if table_exists(conn, "options_cache"):
                try:
                    cached = _options_cache_lookup(conn, mode, as_of, brand, series)
                    if cached is not None:
                        return cached
                except Exception as e:
                    print("options_cache lookup miss:", e)  # fall through to live
            return compute_options(conn, brand, series, as_of=as_of, mode=mode)
    except FileNotFoundError:
        return {"brands": [], "series": [], "models": []}

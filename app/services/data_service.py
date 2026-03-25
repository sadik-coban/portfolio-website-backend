"""
Data Service — reads car_listings.parquet from S3, queries with DuckDB.
Caches the downloaded parquet locally to avoid re-downloading on every request.
"""
import os
import json
import time
import tempfile
import duckdb
from typing import Optional
from app.core.s3_client import download_to_tempfile, object_exists

# ── Cache for the parquet file ──────────────────────────────────────
_cached_parquet_path: str | None = None
_cached_parquet_time: float = 0
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_parquet_path() -> str:
    """Download car_listings.parquet from S3 with TTL caching."""
    global _cached_parquet_path, _cached_parquet_time

    now = time.time()
    if _cached_parquet_path and os.path.exists(_cached_parquet_path) and (now - _cached_parquet_time) < CACHE_TTL_SECONDS:
        return _cached_parquet_path

    # Clean up old temp
    if _cached_parquet_path and os.path.exists(_cached_parquet_path):
        try:
            os.unlink(_cached_parquet_path)
        except OSError:
            pass

    _cached_parquet_path = download_to_tempfile("car_listings.parquet", suffix=".parquet")
    _cached_parquet_time = now
    return _cached_parquet_path


def build_filter_clause(brand, series, min_price, max_price, min_year, max_year, min_km, max_km, fuel, include_series=True):
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


def get_dashboard_data(
    brand: Optional[str] = None, series: Optional[str] = None,
    min_price: Optional[float] = None, max_price: Optional[float] = None,
    min_year: Optional[int] = None, max_year: Optional[int] = None,
    min_km: Optional[int] = None, max_km: Optional[int] = None,
    fuel: Optional[str] = None
):
    base_where, base_params = build_filter_clause(brand, None, min_price, max_price, min_year, max_year, min_km, max_km, fuel, include_series=False)
    final_where, final_params = build_filter_clause(brand, series, min_price, max_price, min_year, max_year, min_km, max_km, fuel, include_series=True)

    try:
        parquet_file = _get_parquet_path()
    except FileNotFoundError:
        return {"error": "car_listings.parquet not found in S3 bucket"}

    conn = duckdb.connect(':memory:')
    try:
        # Lists Query
        lists_query = f"""
            SELECT 
                list(DISTINCT brand ORDER BY brand) FILTER (WHERE brand IS NOT NULL) as brands,
                list(DISTINCT series ORDER BY series) FILTER (WHERE series IS NOT NULL) as series_list 
            FROM read_parquet('{parquet_file}') WHERE {base_where}
        """
        lists_result = conn.execute(lists_query, base_params).fetchone()
        unique_brands = sorted(lists_result[0]) if lists_result and lists_result[0] else []
        unique_series = sorted(lists_result[1]) if lists_result and lists_result[1] else []

        damage_select = f"""
            [{{"part": 'Kaput', "value": SUM(COALESCE(kaput_degisen,0) + COALESCE(kaput_boyali,0))}},
             {{"part": 'Tavan', "value": SUM(COALESCE(tavan_degisen,0) + COALESCE(tavan_boyali,0))}},
             {{"part": 'Bagaj', "value": SUM(COALESCE(bagaj_degisen,0) + COALESCE(bagaj_boyali,0))}},
             {{"part": 'Sol Ön Çamurluk', "value": SUM(COALESCE(fender_fl_degisen,0) + COALESCE(fender_fl_boyali,0))}},
             {{"part": 'Sağ Ön Çamurluk', "value": SUM(COALESCE(fender_fr_degisen,0) + COALESCE(fender_fr_boyali,0))}},
             {{"part": 'Sol Ön Kapı', "value": SUM(COALESCE(door_fl_degisen,0) + COALESCE(door_fl_boyali,0))}},
             {{"part": 'Sağ Ön Kapı', "value": SUM(COALESCE(door_fr_degisen,0) + COALESCE(door_fr_boyali,0))}},
             {{"part": 'Sol Arka Kapı', "value": SUM(COALESCE(door_rl_degisen,0) + COALESCE(door_rl_boyali,0))}},
             {{"part": 'Sağ Arka Kapı', "value": SUM(COALESCE(door_rr_degisen,0) + COALESCE(door_rr_boyali,0))}},
             {{"part": 'Sol Arka Çamurluk', "value": SUM(COALESCE(fender_rl_degisen,0) + COALESCE(fender_rl_boyali,0))}},
             {{"part": 'Sağ Arka Çamurluk', "value": SUM(COALESCE(fender_rr_degisen,0) + COALESCE(fender_rr_boyali,0))}}]::JSON
        """

        # Create a view for filtered data to reuse across queries
        conn.execute(f"CREATE TEMP VIEW filtered_data AS SELECT * FROM read_parquet('{parquet_file}') WHERE {final_where}", final_params)

        # KPI stats
        kpi = conn.execute("SELECT COUNT(*) as total_count, COALESCE(AVG(price), 0) as avg_price FROM filtered_data").fetchone()
        if not kpi or kpi[0] == 0:
            return empty_dashboard_response(unique_brands, unique_series)

        # Boxplot: {brand: [prices]} — same as old json_object_agg(brand, prices)
        boxplot_rows = conn.execute("SELECT brand, list(price) as prices FROM filtered_data WHERE price IS NOT NULL GROUP BY brand").fetchall()
        boxplot_data = {row[0]: row[1] for row in boxplot_rows}

        # Scatter: {brand: [[km, price], ...]} — same as old json_object_agg(brand, points)
        scatter_rows = conn.execute("SELECT brand, list([kb_mileage, price]) as points FROM filtered_data WHERE kb_mileage IS NOT NULL AND price IS NOT NULL GROUP BY brand").fetchall()
        scatter_data = {row[0]: row[1] for row in scatter_rows}

        # Line chart: {years: [...], prices: [...]}
        line_rows = conn.execute("SELECT kb_year, AVG(price) as avg_price FROM filtered_data WHERE kb_year IS NOT NULL GROUP BY kb_year ORDER BY kb_year").fetchall()
        line_chart_data = {
            "years": [row[0] for row in line_rows],
            "prices": [float(row[1]) for row in line_rows],
        }

        # Donut: [{name, value}, ...]
        donut_rows = conn.execute("SELECT kb_fuel, COUNT(*) as count FROM filtered_data WHERE kb_fuel IS NOT NULL GROUP BY kb_fuel").fetchall()
        donut_data = [{"name": row[0], "value": row[1]} for row in donut_rows]

        # Damage chart: [{part, value}, ...]
        damage_query = f"""
            SELECT {damage_select} as data FROM filtered_data
        """
        damage_result = conn.execute(damage_query).fetchone()
        damage_data = json.loads(damage_result[0]) if damage_result and damage_result[0] else []

        # Radar chart: {indicators: [...], series: [...]}
        brand_count_result = conn.execute("SELECT COUNT(DISTINCT brand) FROM filtered_data").fetchone()
        brand_count = brand_count_result[0] if brand_count_result else 0

        if brand_count <= 5:
            global_max = conn.execute("SELECT MAX(price), MAX(kb_mileage), MAX(power_hp_val), MAX(kb_fuel_cons_avg) FROM filtered_data").fetchone()
            radar_rows = conn.execute("SELECT brand, AVG(price), AVG(kb_mileage), AVG(power_hp_val), AVG(kb_fuel_cons_avg) FROM filtered_data GROUP BY brand").fetchall()
            radar_data = {
                "indicators": [
                    {"name": "Fiyat", "max": float(global_max[0] or 0) * 1.1},
                    {"name": "KM", "max": float(global_max[1] or 0) * 1.1},
                    {"name": "Beygir Gücü", "max": float(global_max[2] or 0) * 1.1},
                    {"name": "Yakıt Tüketimi", "max": float(global_max[3] or 0) * 1.1},
                ],
                "series": [
                    {"name": row[0], "value": [float(row[1] or 0), float(row[2] or 0), float(row[3] or 0), float(row[4] or 0)]}
                    for row in radar_rows
                ],
            }
        else:
            radar_data = {"indicators": [], "series": []}

        return {
            "brands": unique_brands,
            "seriesList": unique_series,
            "kpi": {
                "total": kpi[0] or 0,
                "avgPrice": float(kpi[1]) if kpi[1] else 0,
            },
            "boxplotData": boxplot_data,
            "scatterData": scatter_data,
            "lineChartData": line_chart_data,
            "donutChartData": donut_data,
            "damageChartData": damage_data,
            "radarChartData": radar_data,
        }
    except Exception as e:
        print("Dashboard SQL Error:", str(e))
        return empty_dashboard_response([], [])
    finally:
        conn.close()

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

def get_dropdown_options(brand: Optional[str] = None, series: Optional[str] = None):
    try:
        parquet_file = _get_parquet_path()
    except FileNotFoundError:
        return {"brands": [], "series": [], "models": []}
        
    response = {"brands": [], "series": [], "models": []}
    conn = duckdb.connect(':memory:')
    try:
        brands_query = f"SELECT DISTINCT brand FROM read_parquet('{parquet_file}') WHERE brand IS NOT NULL ORDER BY brand"
        response["brands"] = [r[0] for r in conn.execute(brands_query).fetchall()]
        
        if brand:
            series_query = f"SELECT DISTINCT series FROM read_parquet('{parquet_file}') WHERE brand = $brand AND series IS NOT NULL ORDER BY series"
            response["series"] = [r[0] for r in conn.execute(series_query, {"brand": brand}).fetchall()]
        
        if brand and series:
            models_query = f"SELECT DISTINCT model FROM read_parquet('{parquet_file}') WHERE brand = $brand AND series = $series AND model IS NOT NULL ORDER BY model"
            response["models"] = [r[0] for r in conn.execute(models_query, {"brand": brand, "series": series}).fetchall()]
    finally:
        conn.close()
            
    return response

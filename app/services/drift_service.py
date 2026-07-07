"""Drift Service — distribution drift between two car_listings snapshots (DuckDB).

The live path (`get_data_drift`) compares two listing time-slices from cars.duckdb
using KS-test + Earth Mover's Distance on numeric features. Two RAM/speed measures:
  • only the columns the UI shows are pulled (DRIFT_FEATURES) — not all 117 columns;
  • results are cached as JSON on the volume, so a repeat view reads a small file
    instead of re-materializing snapshots + recomputing KS/EMD.
scipy is imported lazily so a process that never runs a drift comparison doesn't pay it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from app.core.config import settings

# The numeric features the drift UI actually renders (app/_site/drift/FinalDrift.tsx
# KEY_FEATURES). Selecting only these — not car_listings' 117 columns — shrinks the
# materialized DataFrame ~10x → far less RAM + faster KS/EMD. Keep in sync with the UI.
DRIFT_FEATURES = [
    "price", "gb_year", "gb_mileage", "power_hp_val", "engine_cc_val", "torque_nm",
    "count_painted", "count_changed", "count_local_painted", "tramer_fee",
]

_CACHE_DIR = Path(settings.VOLUME_DIR) / "drift_cache"


def _cache_path(ref, curr, mode: str, brand: str | None) -> Path:
    key = f"{mode}__{ref}__{curr}__{brand or 'all'}".lower()
    key = re.sub(r"[^a-z0-9_.-]", "_", key)  # filesystem-safe
    return _CACHE_DIR / f"{key}.json"


def _write_cache(path: Path, results) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f)
        os.replace(tmp, path)  # atomic
    except Exception as e:
        print(f"drift cache write skipped ({e}).")


def compute_histogram_bins(data1, data2, bins=20):
    combined = np.concatenate([data1, data2])
    min_val, max_val = np.min(combined), np.max(combined)
    if min_val == max_val:
        return []
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    chart_data = []
    for i in range(bins):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        chart_data.append({
            "bin": round(bin_center, 2),
            "ref_density": float(hist1[i]),
            "curr_density": float(hist2[i]),
        })
    return chart_data


def get_data_drift(ref, curr, mode: str = "specific", brand: str | None = None):
    """Distribution drift between two LISTING snapshots (data, not model).

    Compares car_listings time-slices `ref` vs `curr` (snapshot dates) using KS + EMD
    over DRIFT_FEATURES. Results are cached as JSON on the volume: a computed pair is
    read back from disk on repeat (no SELECT/.df()/KS/EMD, negligible RAM) and survives
    restarts. Historical snapshot pairs are immutable, so cache entries don't go stale.
    """
    cache = _cache_path(ref, curr, mode, brand)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass  # corrupt/partial cache → recompute

    from app.services.data_service import _time_source  # avoid import cycle at module load
    from app.core.db import get_db_connection

    cols = ", ".join(DRIFT_FEATURES)

    def _slice(conn, when):
        src, params = _time_source(when, mode)
        sql = f"SELECT {cols}, brand FROM {src} AS s"
        if brand:
            # brand values are lowercase (audi/bmw); match case-insensitively.
            sql += " WHERE lower(brand) = lower($brand)"
            params = {**params, "brand": brand}
        return conn.execute(sql, params).df()

    with get_db_connection() as conn:
        ref_df = _slice(conn, ref)
        curr_df = _slice(conn, curr)

    if ref_df.empty or curr_df.empty:
        raise FileNotFoundError("No data for one or both snapshots (check date/brand).")

    results = calculate_custom_drift(ref_df, curr_df)
    _write_cache(cache, results)
    return results


def calculate_custom_drift(ref_df: "pd.DataFrame", curr_df: "pd.DataFrame"):  # noqa: F821
    from scipy.stats import ks_2samp, wasserstein_distance  # lazy: only when drift runs
    drift_results = []
    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    if "price" in numeric_cols:
        numeric_cols.remove("price")
        numeric_cols.insert(0, "price")

    for col in numeric_cols:
        ref_data = ref_df[col].dropna().values
        curr_data = curr_df[col].dropna().values
        if len(ref_data) == 0 or len(curr_data) == 0:
            continue

        ks_stat, p_value = ks_2samp(ref_data, curr_data)
        emd_score = wasserstein_distance(ref_data, curr_data)

        std_dev = np.std(ref_data)
        normalized_emd = (emd_score / std_dev) if std_dev > 0 else (999.0 if emd_score > 0 else 0.0)
        is_drifted = (p_value < 0.05) and (normalized_emd > 0.1)
        chart_data = compute_histogram_bins(ref_data, curr_data)

        drift_results.append({
            "feature": col,
            "drift_detected": bool(is_drifted),
            "p_value": float(round(p_value, 5)),
            "ks_statistic": float(round(ks_stat, 4)),
            "emd_score": float(round(emd_score, 2)),
            "chart_data": chart_data,
            "normalized_emd": float(round(normalized_emd, 3)),
        })
    return drift_results

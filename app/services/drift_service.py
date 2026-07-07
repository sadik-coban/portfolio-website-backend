"""Drift Service — distribution drift between two car_listings snapshots (DuckDB).

The live path (`get_data_drift`) compares two listing time-slices from cars.duckdb
using KS-test + Earth Mover's Distance on numeric features. scipy is imported
lazily so a process that never runs a drift comparison doesn't pay its import RAM.
"""
from __future__ import annotations

import numpy as np  # cheap + already resident (bi_service/lgb_service load it at startup)


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

    Compares car_listings time-slices `ref` vs `curr` (snapshot dates), using the
    same time-source as the dashboard. Reuses calculate_custom_drift (KS + EMD).
    """
    from app.services.data_service import _time_source  # avoid import cycle at module load
    from app.core.db import get_db_connection

    def _slice(conn, when):
        src, params = _time_source(when, mode)
        sql = f"SELECT * FROM {src} AS s"
        if brand:
            # brand values are lowercase (audi/bmw); match case-insensitively
            # so a UI sending 'BMW' still filters correctly.
            sql += " WHERE lower(brand) = lower($brand)"
            params = {**params, "brand": brand}
        df = conn.execute(sql, params).df()
        # Drop identity/surrogate columns so they aren't treated as drift features.
        return df.drop(columns=[c for c in ("id", "ad_id") if c in df.columns])

    with get_db_connection() as conn:
        ref_df = _slice(conn, ref)
        curr_df = _slice(conn, curr)

    if ref_df.empty or curr_df.empty:
        raise FileNotFoundError("No data for one or both snapshots (check date/brand).")
    return calculate_custom_drift(ref_df, curr_df)


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

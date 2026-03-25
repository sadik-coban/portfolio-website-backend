"""
Drift Service — reads train_data.parquet from S3.
"""
import os
import pandas as pd
import numpy as np
from app.core.s3_client import download_to_tempfile
from scipy.stats import ks_2samp, wasserstein_distance

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

def get_drift_data(ref_ver: str, curr_ver: str, brand: str | None = None):
    """Download train_data.parquet from S3 for both versions, then compute drift."""
    ref_key = f"{ref_ver}/train_data.parquet"
    curr_key = f"{curr_ver}/train_data.parquet"

    ref_path = download_to_tempfile(ref_key, suffix=".parquet")
    curr_path = download_to_tempfile(curr_key, suffix=".parquet")

    try:
        ref_df = pd.read_parquet(ref_path)
        curr_df = pd.read_parquet(curr_path)
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)

    # Filter by brand if provided
    if brand:
        if "brand" in ref_df.columns:
            ref_df = ref_df[ref_df["brand"] == brand]
        if "brand" in curr_df.columns:
            curr_df = curr_df[curr_df["brand"] == brand]
        if ref_df.empty or curr_df.empty:
            raise FileNotFoundError(f"No data found for brand '{brand}' in one or both versions.")
    
    return calculate_custom_drift(ref_df, curr_df)


def calculate_custom_drift(ref_df: pd.DataFrame, curr_df: pd.DataFrame):
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

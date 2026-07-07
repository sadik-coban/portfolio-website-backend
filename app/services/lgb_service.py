"""Price prediction — the best model only (LightGBM · TF-IDF+SVD).

The serving artifact (`serving/lightgbm_tfidf_svd.pkl`) is downloaded from S3 on
startup and held in memory (the model is NOT committed to git). The pickle bundles
the trained LightGBM model plus all preprocessing:
  { model, tfidf:{model,series}:{vec,svd,n}, cat_maps, feat_cols, CAT, NUM, TEXT }

Serving flow (mirrors the training pipeline):
  categoricals → cat_maps label-encode · numerics as-is ·
  model/series text → TF-IDF vec → SVD embedding (model_0..149, series_0..19) →
  assemble in feat_cols order → model.predict → np.expm1(clip(pred, 0, log1p(1.5e7))).
"""
import os
import pickle
import threading

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.s3_client import download_to_tempfile

_BUNDLE = None
_LOCK = threading.Lock()

# CAT/NUM come from the bundle; these are the accepted API input fields.
CAT_FIELDS = ["brand", "kb_body_type", "kb_drivetrain", "segment", "kb_transmission", "kb_fuel"]
NUM_FIELDS = ["vehicle_age", "gb_mileage", "power_hp_val", "engine_cc_val",
              "count_painted", "count_changed", "count_local_painted", "is_heavy_damaged"]
MARGIN_PCT = 6.6  # OOF MAPE of the LightGBM model → shown as a ± band


def load_model(force: bool = False):
    """Load the serving bundle into memory (download from S3 once). Thread-safe."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None and not force:
            return _BUNDLE
        path = download_to_tempfile(settings.SERVING_MODEL_KEY, suffix=".pkl")
        try:
            with open(path, "rb") as f:
                _BUNDLE = pickle.load(f)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return _BUNDLE


def unload_model():
    global _BUNDLE
    _BUNDLE = None


def _embed(bundle, col: str, text: str) -> np.ndarray:
    e = bundle["tfidf"][col]
    x = e["vec"].transform([text or ""])
    x = e["svd"].transform(x)
    return np.asarray(x)[0]


def predict_price(inp: dict) -> dict:
    """inp: dict with CAT_FIELDS + NUM_FIELDS + 'model' + 'series'. Returns price + band."""
    b = load_model()
    cm, feat = b["cat_maps"], b["feat_cols"]
    row: dict = {}
    for c in b["CAT"]:
        m = cm[c]
        v = str(inp.get(c, "")).strip()
        key = v.lower() if c == "brand" else v
        row[c] = m.get(key, -1)  # unseen category → -1
    for c in b["NUM"]:
        try:
            row[c] = float(inp.get(c) if inp.get(c) not in (None, "") else 0)
        except (TypeError, ValueError):
            row[c] = 0.0
    me = _embed(b, "model", str(inp.get("model", "")))
    se = _embed(b, "series", str(inp.get("series", "")))
    for i in range(len(me)):
        row[f"model_{i}"] = float(me[i])
    for i in range(len(se)):
        row[f"series_{i}"] = float(se[i])

    X = pd.DataFrame([[row[c] for c in feat]], columns=feat)
    raw = float(b["model"].predict(X)[0])
    price = float(np.expm1(np.clip(raw, 0.0, np.log1p(1.5e7))))
    price = int(round(price))
    lo = int(round(price * (1 - MARGIN_PCT / 100)))
    hi = int(round(price * (1 + MARGIN_PCT / 100)))
    return {
        "price": price,
        "price_range": {"min": lo, "max": hi, "margin_percent": MARGIN_PCT},
        "model": "LightGBM · TF-IDF+SVD",
        "currency": "TL",
    }

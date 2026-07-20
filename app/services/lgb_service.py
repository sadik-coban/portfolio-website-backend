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

from app.core.config import settings
from app.core.s3_client import download_to_tempfile

_BUNDLE = None
_LOCK = threading.Lock()

# CAT/NUM come from the bundle; these mirror it as the accepted API input fields.
# Panel damage (2026-07-13 model onward): single panels carry a state, multi-panel
# groups carry per-operation counts. Anything the caller omits falls back to 0 (NUM)
# or an unseen-category -1 (CAT), so a stale caller silently loses the damage signal
# rather than erroring — keep these in step with app/models/schemas.py.
CAT_FIELDS = ["brand", "kb_body_type", "kb_drivetrain", "segment", "kb_transmission", "kb_fuel",
              "roof_state", "hood_state", "trunk_state"]
NUM_FIELDS = ["vehicle_age", "gb_mileage", "power_hp_val", "engine_cc_val",
              "door_changed", "door_painted", "door_local",
              "fender_changed", "fender_painted", "fender_local",
              "bumper_changed", "bumper_painted", "bumper_local", "is_heavy_damaged"]
MARGIN_PCT = 6.6  # OOF MAPE of the LightGBM model → shown as a ± band


def _assert_bundle_matches_schema(bundle) -> None:
    """Fail loudly if a bundle wants inputs the request schema no longer carries.

    Only the CAT/NUM columns are checked — those are read straight off the request
    (`inp.get(col)`), so a name the schema dropped silently becomes 0.0 / None. TEXT and
    the derived model_*/series_* SVD dimensions are built server-side and aren't affected.
    """
    from app.models.schemas import PricePredictInput

    fields = set(PricePredictInput.model_fields)
    wanted = set(bundle.get("CAT", [])) | set(bundle.get("NUM", []))
    missing = sorted(wanted - fields)
    if missing:
        raise RuntimeError(
            "Serving bundle expects request fields that PricePredictInput no longer has: "
            f"{missing}. This bundle predates the panel-level damage schema; serving it "
            "would default those inputs and return confident wrong prices. Point "
            "SERVING_MODEL_KEY at a current bundle, or restore the fields to the schema."
        )


def _load_pickle():
    """Combined pickle bundle: { model, tfidf, cat_maps, feat_cols, CAT, NUM, TEXT }.

    This is the rollback path (clear SERVING_MODEL_TXT_KEY/SERVING_ENCODERS_KEY). It is
    guarded because an OLD bundle expects the retired count_painted/count_changed/
    count_local_painted numerics: those no longer exist on PricePredictInput, so every
    one would resolve to 0.0 and the model would price a repaired car as undamaged —
    HTTP 200, no error, wrong number. A rollback that can't be exercised safely has to
    refuse to load rather than answer confidently.
    """
    path = download_to_tempfile(settings.SERVING_MODEL_KEY, suffix=".pkl")
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        _assert_bundle_matches_schema(bundle)
        return bundle
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _load_native():
    """Version-resilient: native LightGBM booster (.txt) + encoders pickle → same bundle
    shape as the combined pickle (`encoders.pkl` holds everything except the model)."""
    import lightgbm as lgb
    txt = download_to_tempfile(settings.SERVING_MODEL_TXT_KEY, suffix=".txt")
    enc = download_to_tempfile(settings.SERVING_ENCODERS_KEY, suffix=".pkl")
    try:
        booster = lgb.Booster(model_file=txt)
        with open(enc, "rb") as f:
            encoders = pickle.load(f)
        # feat_cols order is preserved when assembling X → the Booster predicts positionally.
        return {"model": booster, **encoders}
    finally:
        for p in (txt, enc):
            try:
                os.unlink(p)
            except OSError:
                pass


def load_model(force: bool = False):
    """Load the serving bundle into memory (download from S3 once). Thread-safe.
    Prefers the native (.txt + encoders.pkl) pair when both keys are configured;
    otherwise falls back to the combined pickle (unchanged default)."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None and not force:
            return _BUNDLE
        use_native = bool(settings.SERVING_MODEL_TXT_KEY and settings.SERVING_ENCODERS_KEY)
        _BUNDLE = _load_native() if use_native else _load_pickle()
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

    # Keep the named DataFrame the model was trained with — identical output to the
    # original serving code (verified). Import pandas lazily: LightGBM already pulls it
    # in, so this adds no startup RAM while keeping the serving module import cheap.
    import pandas as pd
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

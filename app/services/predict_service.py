"""
Predict Service — loads models and registry from the persistent volume.

Layout (settings.VOLUME_DIR):
    registry.json
    <version_id>/model.cbm | metrics.json | shap_summary.png | train_data.parquet

Models/registry are populated by the admin API (POST /admin/models/sync-s3),
which pulls them from the curated S3 bucket onto the volume.
"""
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from app.core.config import settings
from app.models.schemas import CarPredictionInput, DamageInputs

loaded_models = {}


def _volume() -> Path:
    return Path(settings.VOLUME_DIR)


def registry_path() -> Path:
    return _volume() / "registry.json"


def model_path(version_id: str) -> Path:
    return _volume() / version_id / "model.cbm"


def get_versions():
    """Read registry.json from the volume (newest first)."""
    path = registry_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(data, key=lambda x: x.get("date", ""), reverse=True)
    except Exception as e:
        print(f"Version read error from volume: {e}")
        return []


def preload_latest_model():
    # No-op when the RAM cache is disabled — models load on demand per request.
    if not settings.MODEL_CACHE_ENABLED:
        print("Model cache disabled (MODEL_CACHE_ENABLED=False); skipping preload.")
        return
    versions = get_versions()
    if versions:
        v_id = versions[0]["version_id"]
        load_model_to_memory(v_id)


def _load_model_from_disk(version_id: str) -> CatBoostRegressor:
    path = model_path(version_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found on volume: {path}. "
            f"Pull it first via POST /admin/models/sync-s3."
        )
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_model_to_memory(version_id: str):
    """Return a model for `version_id`.

    With MODEL_CACHE_ENABLED=False (current default) the model is loaded fresh
    from the volume every call and NOT retained — nothing stays resident in
    `loaded_models`. With caching on, it is loaded once and reused.
    """
    if not settings.MODEL_CACHE_ENABLED:
        return _load_model_from_disk(version_id)

    if version_id not in loaded_models:
        loaded_models[version_id] = _load_model_from_disk(version_id)
        print(f"Model {version_id} loaded into memory from volume.")
    return loaded_models[version_id]


def unload_model(version_id: str):
    """Evict a single model from the in-memory cache (after delete)."""
    loaded_models.pop(version_id, None)


def unload_models():
    loaded_models.clear()

def calculate_risk_score_logic(damage: DamageInputs) -> float:
    score = 0.0
    if damage.roof_status == "Değişen": score += 150
    elif damage.roof_status == "Boyalı": score += 75
    elif damage.roof_status == "Lokal Boyalı": score += 40

    if damage.hood_status == "Değişen": score += 60
    elif damage.hood_status == "Boyalı": score += 30
    elif damage.hood_status == "Lokal Boyalı": score += 15

    if damage.trunk_status == "Değişen": score += 40
    elif damage.trunk_status == "Boyalı": score += 20
    elif damage.trunk_status == "Lokal Boyalı": score += 10

    score += (damage.doors_changed * 10) + (damage.doors_painted * 5) + (damage.doors_local * 2)
    score += (damage.fenders_changed * 8) + (damage.fenders_painted * 4) + (damage.fenders_local * 2)
    return float(score)

def predict_price(version_id: str, input_data: CarPredictionInput):
    model = load_model_to_memory(version_id)
    calculated_score = calculate_risk_score_logic(input_data.damage_details)

    feature_dict = {
        "brand": str(input_data.brand),
        "series": str(input_data.series),
        "model": str(input_data.model),
        "engine_cc_val": float(input_data.engine_cc_val),
        "power_hp_val": float(input_data.power_hp_val),
        "kb_drivetrain": str(input_data.kb_drivetrain),
        "gb_warranty_status": str(input_data.gb_warranty_status),
        "torque_nm": float(input_data.torque_nm),
        "cylinder_count": int(input_data.cylinder_count),
        "is_heavy_damaged": int(input_data.is_heavy_damaged),
        "year": int(input_data.year),
        "mileage": float(input_data.mileage),
        "transmission": str(input_data.transmission),
        "fuel": str(input_data.fuel),
        "body_type": str(input_data.body_type),
        "segment_clean": str(input_data.segment_clean),
        "expert_risk_score": float(calculated_score),
    }

    df = pd.DataFrame([feature_dict])
    expected_features = model.feature_names_
    df = df[expected_features]

    cat_features = [
        "brand", "series", "model", "transmission", "fuel", 
        "body_type", "kb_drivetrain", "gb_warranty_status", "segment_clean"
    ]
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    prediction_result = model.predict(df)
    q05 = prediction_result[0][0]
    q50 = prediction_result[0][1]
    q95 = prediction_result[0][2]

    q05 = q05 * 0.98
    q95 = q95 * 1.02
    implied_margin_percent = ((q95 - q05) / 2) / q50 * 100

    return {
        "price": int(q50),
        "price_range": {
            "min": int(q05),
            "max": int(q95),
            "margin_percent": round(implied_margin_percent, 1),
        },
        "version": version_id,
        "calculated_risk_score": calculated_score,
        "currency": "TL",
    }

from datetime import date
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import DriftResponse, PricePredictInput
from app.services import lgb_service
from app.services.data_service import get_snapshots
from app.services.drift_service import get_data_drift
from app.services.bi_service import get_bi_meta, get_bi_agg

router = APIRouter()


# ─── Prediction ──────────────────────────────────────────────────────

@router.post(
    "/api/predict",
    tags=["Prediction"],
    summary="Predict car price (best model)",
    description=(
        "Predict a used-car price with the best model (LightGBM · TF-IDF+SVD). The model "
        "+ preprocessing are downloaded from S3 on startup and held in memory. Returns a "
        "point estimate + a ±MAPE band. Categoricals use the training vocabulary (Turkish); "
        "model + series are free text (TF-IDF+SVD embedded server-side)."
    ),
    responses={500: {"description": "Prediction error (model unavailable or bad input)"}},
)
def api_predict(input_data: PricePredictInput):
    """Single-model price prediction from the in-memory LightGBM bundle."""
    try:
        return lgb_service.predict_price(input_data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Serving model unavailable: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Drift Detection ────────────────────────────────────────────────

@router.get(
    "/api/data-drift",
    tags=["Drift Detection"],
    summary="Data drift between two listing snapshots (not model)",
    description=(
        "Compares the LISTING data distribution between two snapshot dates "
        "(car_listings time-slices) using KS-test + Earth Mover's Distance on numeric "
        "features. Use /api/snapshots for valid dates. mode='specific' compares the raw "
        "snapshots; mode='until' compares point-in-time states."
    ),
    response_model=DriftResponse,
    responses={404: {"description": "No data for one or both snapshots"}, 500: {"description": "Analysis error"}},
)
def analyze_data_drift(
    ref: date = Query(..., description="Reference snapshot date"),
    curr: date = Query(..., description="Current snapshot date to compare"),
    mode: Literal["specific", "until"] = Query("specific", description="'specific' = raw snapshot; 'until' = point-in-time as-of"),
    brand: Optional[str] = Query(None, description="Filter to a specific brand (e.g. 'bmw')"),
):
    """Statistical drift analysis between two listing-data snapshots."""
    try:
        results = get_data_drift(ref, curr, mode=mode, brand=brand)
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz Hatası: {str(e)}")


# ─── Dashboard & Analytics ───────────────────────────────────────────

@router.get(
    "/api/snapshots",
    tags=["Dashboard & Analytics"],
    summary="List available data snapshot dates",
    description="Returns the scrape (snapshot) dates with per-snapshot and cumulative (until) listing counts, for the dashboard time selector.",
)
def api_get_snapshots():
    return get_snapshots()


@router.get(
    "/api/bi/meta",
    tags=["Dashboard & Analytics"],
    summary="BI dashboard metadata + label dictionaries",
    description=(
        "Static metadata for the client dashboard: meta (counts, brands, fuels, segments, "
        "snapshots, tramer) + the label dictionaries (series/city/district/model/body) used "
        "for the filter dropdowns and chart labels. Fetched once. Derived from cars.duckdb "
        "server-side — the ~30K raw rows never leave the backend."
    ),
)
def api_get_bi_meta():
    """Metadata + dicts for the backend-fed BI dashboard."""
    try:
        return get_bi_meta()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/api/bi/agg",
    tags=["Dashboard & Analytics"],
    summary="BI dashboard aggregation for the current filter selection",
    description=(
        "Server-side aggregation for the rich BI dashboard (KPIs, segment/price/fuel/damage/"
        "location charts, Turkey map + districts). Filters, the damage-type toggle, and the "
        "selected province are applied server-side so no row-level data is shipped to the client. "
        "Returns { agg, districts, damage }."
    ),
)
def api_get_bi_agg(
    brand: int = Query(-1, description="Brand index (0 BMW, 1 Audi; -1 = any)"),
    series: int = Query(-1, description="Series dict index (-1 = any)"),
    fuel: int = Query(-1, description="Fuel index (0 Benzin, 1 Dizel, 2 LPG, 3 Hibrit; -1 = any)"),
    seg: int = Query(-1, description="Segment index into B,C,D,E,F,S (-1 = any)"),
    damage: int = Query(-1, description="Damage: -1 any, 0 clean, 1 heavy-damaged"),
    yearMin: Optional[int] = Query(None),
    yearMax: Optional[int] = Query(None),
    priceMin: Optional[int] = Query(None, description="Min price (TL)"),
    priceMax: Optional[int] = Query(None, description="Max price (TL)"),
    kmMin: Optional[int] = Query(None),
    kmMax: Optional[int] = Query(None),
    dmgType: int = Query(-1, description="Damage-type toggle: -1 any, 1 local, 2 painted, 3 changed"),
    province: Optional[str] = Query(None, description="Selected province (city) for the district drilldown"),
):
    """Filtered + aggregated BI payload (parity with the old client-side compute.ts)."""
    filters = {
        "brand": brand, "series": series, "fuel": fuel, "seg": seg, "damage": damage,
        "yearMin": yearMin, "yearMax": yearMax, "priceMin": priceMin, "priceMax": priceMax,
        "kmMin": kmMin, "kmMax": kmMax,
    }
    try:
        return get_bi_agg(filters, dmg_type=dmgType, province=province)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

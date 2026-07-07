import io
from datetime import date
from pathlib import Path as FsPath
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.models.schemas import CarPredictionInput, PredictionResponse, DriftResponse, DashboardResponse, DropdownOptionsResponse, VersionInfo, PricePredictInput
from app.services.predict_service import get_versions, predict_price
from app.services import lgb_service
from app.services.data_service import (
    get_dashboard_data, get_dropdown_options, get_snapshots, get_price_history,
)
from app.services.drift_service import get_drift_data, get_data_drift
from app.services.bi_service import get_bi_meta, get_bi_agg

router = APIRouter()

# ─── Model Management ───────────────────────────────────────────────

@router.get(
    "/versions",
    tags=["Model Management"],
    summary="List all model versions",
    description="Returns a list of all registered model versions sorted by date (newest first). Each entry includes version ID, training date, and metadata.",
    response_model=list[VersionInfo],
)
def api_get_versions():
    """Retrieve all available model versions from the registry."""
    return get_versions()


@router.get(
    "/api/shap/{version_id}",
    tags=["Model Management"],
    summary="Get SHAP summary plot",
    description="Returns the SHAP feature importance summary plot (PNG image) for a given model version.",
    responses={
        200: {"content": {"image/png": {}}, "description": "SHAP summary plot image"},
        404: {"description": "SHAP plot not found for the specified version"},
    },
)
def get_shap_plot(
    version_id: str = Path(..., description="Model version ID (e.g. 'v12')"),
):
    """Serve the SHAP summary plot image from the volume."""
    path = FsPath(settings.VOLUME_DIR) / version_id / "shap_summary.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SHAP grafiği bulunamadı.")
    return FileResponse(str(path), media_type="image/png")


# ─── Prediction ──────────────────────────────────────────────────────

@router.post(
    "/predict/{version_id}",
    tags=["Prediction"],
    summary="Predict car price",
    description=(
        "Predict the price of a second-hand car using a specified model version. "
        "Returns the median predicted price (Q50) along with a confidence interval (Q5–Q95) "
        "and the calculated expert risk score based on damage inputs."
    ),
    response_model=PredictionResponse,
    responses={
        404: {"description": "Model version not found"},
        500: {"description": "Internal prediction error"},
    },
)
def api_predict_price(
    version_id: str = Path(..., description="Model version ID to use for prediction (e.g. 'v12')"),
    input_data: CarPredictionInput = ...,
):
    """Run price prediction using the specified CatBoost model version."""
    try:
        return predict_price(version_id, input_data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    "/drift/{ref_ver}/{curr_ver}",
    tags=["Drift Detection"],
    summary="Analyze data drift between two versions",
    description=(
        "Compare training datasets of two model versions to detect data drift. "
        "Uses Kolmogorov-Smirnov test and Earth Mover's Distance (Wasserstein) on all numeric features. "
        "Optionally filter by brand to analyze drift within a specific brand segment."
    ),
    response_model=DriftResponse,
    responses={
        404: {"description": "Training data not found for one or both versions"},
        500: {"description": "Analysis error"},
    },
)
def analyze_drift(
    ref_ver: str = Path(..., description="Reference (baseline) model version ID"),
    curr_ver: str = Path(..., description="Current model version ID to compare against the reference"),
    brand: Optional[str] = Query(None, description="Filter drift analysis to a specific car brand (e.g. 'Toyota')"),
):
    """Perform statistical drift analysis between two model training datasets."""
    try:
        results = get_drift_data(ref_ver, curr_ver, brand=brand)
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz Hatası: {str(e)}")


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
    "/api/dashboard-data",
    tags=["Dashboard & Analytics"],
    summary="Get aggregated dashboard data",
    description=(
        "Returns pre-aggregated data for the frontend dashboard including KPIs, "
        "boxplot distributions, scatter data, price trend lines, fuel type donut chart, "
        "radar comparisons, and damage heatmap data. All data can be filtered by brand, series, "
        "price range, year range, mileage range, and fuel type."
    ),
)
def api_get_dashboard(
    brand: Optional[str] = Query(None, description="Filter by car brand (e.g. 'BMW')"),
    series: Optional[str] = Query(None, description="Filter by car series (e.g. '3 Serisi')"),
    min_price: Optional[float] = Query(None, description="Minimum price filter (TL)"),
    max_price: Optional[float] = Query(None, description="Maximum price filter (TL)"),
    min_year: Optional[int] = Query(None, description="Minimum model year filter"),
    max_year: Optional[int] = Query(None, description="Maximum model year filter"),
    min_km: Optional[int] = Query(None, description="Minimum mileage filter (km)"),
    max_km: Optional[int] = Query(None, description="Maximum mileage filter (km)"),
    fuel: Optional[str] = Query(None, description="Filter by fuel type (e.g. 'Benzin', 'Dizel')"),
    as_of: Optional[date] = Query(None, description="Snapshot date (from /api/snapshots). Empty = latest/current."),
    mode: Literal["specific", "until"] = Query("until", description="'specific' = only that snapshot; 'until' = point-in-time up to that date (one row per ad)."),
):
    """Serve aggregated analytics data for the dashboard frontend."""
    return get_dashboard_data(
        brand, series, min_price, max_price,
        min_year, max_year, min_km, max_km, fuel,
        as_of=as_of, mode=mode,
    )


@router.get(
    "/api/snapshots",
    tags=["Dashboard & Analytics"],
    summary="List available data snapshot dates",
    description="Returns the scrape (snapshot) dates with per-snapshot and cumulative (until) listing counts, for the dashboard time selector.",
)
def api_get_snapshots():
    return get_snapshots()


@router.get(
    "/api/price-history/{ad_id}",
    tags=["Dashboard & Analytics"],
    summary="Price/mileage history of a listing across snapshots",
    description="Returns the per-snapshot price and mileage time series (with price deltas) for a single ad_id.",
)
def api_get_price_history(ad_id: int = Path(..., description="Listing ad_id")):
    return get_price_history(ad_id)


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


@router.get(
    "/api/options",
    tags=["Dashboard & Analytics"],
    summary="Get dropdown filter options",
    description=(
        "Returns available dropdown values for brand, series, and model filters. "
        "Series options are filtered by the selected brand, and model options are filtered "
        "by both brand and series — enabling cascading dropdown menus."
    ),
    response_model=DropdownOptionsResponse,
)
def api_get_options(
    brand: Optional[str] = Query(None, description="Selected brand to filter series options"),
    series: Optional[str] = Query(None, description="Selected series to filter model options"),
    as_of: Optional[date] = Query(None, description="Snapshot date (from /api/snapshots). Empty = latest/current."),
    mode: Literal["specific", "until"] = Query("until", description="'specific' = only that snapshot; 'until' = point-in-time up to that date."),
):
    """Provide cascading dropdown options for the filter UI (time-scoped)."""
    return get_dropdown_options(brand, series, as_of=as_of, mode=mode)

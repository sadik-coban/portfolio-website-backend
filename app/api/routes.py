import os
import io
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse, StreamingResponse

from app.core.s3_client import read_s3_bytes
from app.models.schemas import CarPredictionInput, PredictionResponse, DriftResponse, DashboardResponse, DropdownOptionsResponse, VersionInfo
from app.services.predict_service import get_versions, predict_price
from app.services.data_service import get_dashboard_data, get_dropdown_options
from app.services.drift_service import get_drift_data

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
    """Serve the SHAP summary plot image from S3."""
    try:
        image_bytes = read_s3_bytes(f"{version_id}/shap_summary.png")
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="SHAP grafiği bulunamadı.")


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
):
    """Serve aggregated analytics data for the dashboard frontend."""
    return get_dashboard_data(
        brand, series, min_price, max_price, 
        min_year, max_year, min_km, max_km, fuel
    )


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
):
    """Provide cascading dropdown options for the filter UI."""
    return get_dropdown_options(brand, series)

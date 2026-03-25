from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Models ──────────────────────────────────────────────────

class DamageInputs(BaseModel):
    """Body damage details used to calculate the expert risk score."""
    roof_status: str = Field("Orjinal", description="Roof condition: Orjinal, Boyalı, Değişen, or Lokal Boyalı")
    hood_status: str = Field("Orjinal", description="Hood condition: Orjinal, Boyalı, Değişen, or Lokal Boyalı")
    trunk_status: str = Field("Orjinal", description="Trunk condition: Orjinal, Boyalı, Değişen, or Lokal Boyalı")
    doors_changed: int = Field(0, ge=0, le=4, description="Number of doors replaced (0-4)")
    doors_painted: int = Field(0, ge=0, le=4, description="Number of doors repainted (0-4)")
    doors_local: int = Field(0, ge=0, le=4, description="Number of doors with local paint (0-4)")
    fenders_changed: int = Field(0, ge=0, le=4, description="Number of fenders replaced (0-4)")
    fenders_painted: int = Field(0, ge=0, le=4, description="Number of fenders repainted (0-4)")
    fenders_local: int = Field(0, ge=0, le=4, description="Number of fenders with local paint (0-4)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "roof_status": "Orjinal",
                    "hood_status": "Boyalı",
                    "trunk_status": "Orjinal",
                    "doors_changed": 0,
                    "doors_painted": 1,
                    "doors_local": 0,
                    "fenders_changed": 0,
                    "fenders_painted": 0,
                    "fenders_local": 0,
                }
            ]
        }
    }


class CarPredictionInput(BaseModel):
    """Complete input payload for car price prediction."""
    brand: str = Field(..., description="Car brand (e.g. 'Toyota', 'BMW')")
    series: str = Field(..., description="Car series (e.g. 'Corolla', '3 Serisi')")
    model: str = Field(..., description="Car model variant (e.g. '1.6 Vision')")
    transmission: str = Field(..., description="Transmission type (e.g. 'Otomatik', 'Manuel')")
    fuel: str = Field(..., description="Fuel type (e.g. 'Benzin', 'Dizel', 'Hibrit')")
    body_type: str = Field(..., description="Body type (e.g. 'Sedan', 'Hatchback', 'SUV')")
    kb_drivetrain: str = Field(..., description="Drivetrain type (e.g. 'Önden Çekiş', '4WD')")
    gb_warranty_status: str = Field(..., description="Warranty status (e.g. 'Var', 'Yok')")
    segment_clean: str = Field(..., description="Vehicle segment (e.g. 'C-Segment', 'D-Segment')")
    year: int = Field(..., gt=1900, le=2100, description="Model year of the car")
    mileage: float = Field(..., ge=0, description="Total mileage in km")
    engine_cc_val: float = Field(..., description="Engine displacement in cc")
    power_hp_val: float = Field(..., description="Engine power in horsepower")
    torque_nm: float = Field(..., description="Engine torque in Nm")
    cylinder_count: int = Field(..., description="Number of engine cylinders")
    is_heavy_damaged: int = Field(0, description="1 if the car has severe structural damage, 0 otherwise")
    damage_details: DamageInputs = Field(..., description="Detailed body damage inputs for risk scoring")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "Toyota",
                    "series": "Corolla",
                    "model": "1.6 Vision",
                    "transmission": "Otomatik",
                    "fuel": "Benzin",
                    "body_type": "Sedan",
                    "kb_drivetrain": "Önden Çekiş",
                    "gb_warranty_status": "Yok",
                    "segment_clean": "C-Segment",
                    "year": 2020,
                    "mileage": 45000,
                    "engine_cc_val": 1598,
                    "power_hp_val": 132,
                    "torque_nm": 160,
                    "cylinder_count": 4,
                    "is_heavy_damaged": 0,
                    "damage_details": {
                        "roof_status": "Orjinal",
                        "hood_status": "Orjinal",
                        "trunk_status": "Orjinal",
                        "doors_changed": 0,
                        "doors_painted": 0,
                        "doors_local": 0,
                        "fenders_changed": 0,
                        "fenders_painted": 0,
                        "fenders_local": 0,
                    },
                }
            ]
        }
    }


# ─── Response Models ─────────────────────────────────────────────────

class VersionInfo(BaseModel):
    """Metadata for a single model version."""
    version_id: str = Field(..., description="Unique identifier for the model version")
    date: Optional[str] = Field(None, description="Training date (ISO format)")

    model_config = {"extra": "allow"}


class PriceRange(BaseModel):
    """Confidence interval for the predicted price."""
    min: int = Field(..., description="Lower bound price estimate (Q5)")
    max: int = Field(..., description="Upper bound price estimate (Q95)")
    margin_percent: float = Field(..., description="Implied margin percentage based on the price range")


class PredictionResponse(BaseModel):
    """Response payload from the price prediction endpoint."""
    price: int = Field(..., description="Median predicted price in TL (Q50)")
    price_range: PriceRange = Field(..., description="Confidence interval (Q5–Q95)")
    version: str = Field(..., description="Model version used for prediction")
    calculated_risk_score: float = Field(..., description="Expert risk score calculated from damage inputs")
    currency: str = Field("TL", description="Currency of the price")


class HistogramBin(BaseModel):
    """A single histogram bin for drift visualization."""
    bin: float = Field(..., description="Center of the histogram bin")
    ref_density: float = Field(..., description="Density value for the reference dataset")
    curr_density: float = Field(..., description="Density value for the current dataset")


class FeatureDriftResult(BaseModel):
    """Drift analysis result for a single feature."""
    feature: str = Field(..., description="Name of the numeric feature")
    drift_detected: bool = Field(..., description="Whether significant drift was detected")
    p_value: float = Field(..., description="KS-test p-value")
    ks_statistic: float = Field(..., description="KS-test statistic")
    emd_score: float = Field(..., description="Earth Mover's Distance (Wasserstein)")
    normalized_emd: float = Field(..., description="EMD normalized by reference std deviation")
    chart_data: list[HistogramBin] = Field(default=[], description="Histogram data for visualization")


class DriftResponse(BaseModel):
    """Response from the drift analysis endpoint."""
    results: list[FeatureDriftResult] = Field(..., description="Per-feature drift analysis results")


class DropdownOptionsResponse(BaseModel):
    """Cascading dropdown options for the filter UI."""
    brands: list[str] = Field(default=[], description="Available car brands")
    series: list[str] = Field(default=[], description="Available series (filtered by brand)")
    models: list[str] = Field(default=[], description="Available models (filtered by brand + series)")


class DashboardResponse(BaseModel):
    """Aggregated dashboard data for the frontend."""
    brands: list[str] = Field(default=[], description="List of unique brands in the filtered dataset")
    seriesList: list[str] = Field(default=[], description="List of unique series in the filtered dataset")
    kpi: dict = Field(default={}, description="Key performance indicators (total count, average price)")
    boxplotData: dict = Field(default={}, description="Price distribution by brand (boxplot)")
    scatterData: dict = Field(default={}, description="Mileage vs price scatter data by brand")
    lineChartData: dict = Field(default={}, description="Average price trend by year")
    donutChartData: list = Field(default=[], description="Fuel type distribution (donut chart)")
    damageChartData: list = Field(default=[], description="Damage part frequency data")
    radarChartData: dict = Field(default={}, description="Multi-dimensional brand comparison (radar chart)")

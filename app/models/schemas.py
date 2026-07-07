from pydantic import BaseModel, Field


# ─── Price prediction (best model: LightGBM · TF-IDF+SVD) ────────────

class PricePredictInput(BaseModel):
    """Inputs for /api/predict. Categoricals use the training vocabulary (Turkish);
    model + series are free text (TF-IDF+SVD embedded server-side)."""
    brand: str = Field(..., description="bmw | audi")
    series: str = Field(..., description="Series text, e.g. '5 Serisi', 'A4'")
    model: str = Field(..., description="Model text, e.g. '520i Executive M Sport'")
    kb_body_type: str = Field(..., description="Sedan | Hatchback/5 | Hatchback/3 | Coupe | Cabrio | Station wagon | MPV")
    kb_drivetrain: str = Field(..., description="Arkadan İtiş | Önden Çekiş | 4WD (Sürekli) | AWD (Elektronik)")
    segment: str = Field(..., description="B | C | D | E | F | S")
    kb_transmission: str = Field(..., description="Otomatik | Düz | Yarı Otomatik")
    kb_fuel: str = Field(..., description="Benzin | Dizel | LPG & Benzin | Hibrit")
    vehicle_age: int = Field(..., ge=0, le=60)
    gb_mileage: float = Field(..., ge=0)
    power_hp_val: float = Field(..., ge=0)
    engine_cc_val: float = Field(..., ge=0)
    count_painted: int = Field(0, ge=0, description="Number of painted body parts")
    count_changed: int = Field(0, ge=0, description="Number of replaced body parts")
    count_local_painted: int = Field(0, ge=0, description="Number of locally-painted body parts")
    is_heavy_damaged: int = Field(0, ge=0, le=1)


# ─── Drift response (/api/data-drift) ────────────────────────────────

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

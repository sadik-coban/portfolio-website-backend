from pydantic import BaseModel, ConfigDict, Field


# ─── Price prediction (best model: LightGBM · TF-IDF+SVD) ────────────

class PricePredictInput(BaseModel):
    """Inputs for /api/predict. Categoricals use the training vocabulary (Turkish);
    model + series are free text (TF-IDF+SVD embedded server-side).

    Damage is PANEL-level (2026-07-13 model onward): the three single panels carry a
    state, the multi-panel groups carry per-operation counts. The older aggregate
    count_painted / count_changed / count_local_painted fields are gone — the current
    model never sees them.
    """

    # Reject unknown fields instead of ignoring them (pydantic's default). A caller still
    # posting the retired count_painted/count_changed/count_local_painted would otherwise
    # get HTTP 200 with every panel silently defaulted to undamaged — a heavily repaired
    # car quoted at a clean-car price, with no error anywhere. A loud 422 makes that skew
    # impossible to miss. Consequence to plan for: during a deploy the frontend and this
    # schema must move together, or /api/predict 422s until both sides land.
    model_config = ConfigDict(extra="forbid")
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

    # Single panels → state. Vocabulary: original | painted | local | changed
    roof_state: str = Field("original", description="original | painted | local | changed")
    hood_state: str = Field("original", description="original | painted | local | changed")
    trunk_state: str = Field("original", description="original | painted | local | changed")

    # Multi-panel groups → how many panels had each operation.
    # doors: 4 panels · fenders: 4 panels · bumpers: 2 (front, rear)
    door_changed: int = Field(0, ge=0, le=4, description="Doors replaced (of 4)")
    door_painted: int = Field(0, ge=0, le=4, description="Doors painted (of 4)")
    door_local: int = Field(0, ge=0, le=4, description="Doors locally painted (of 4)")
    fender_changed: int = Field(0, ge=0, le=4, description="Fenders replaced (of 4)")
    fender_painted: int = Field(0, ge=0, le=4, description="Fenders painted (of 4)")
    fender_local: int = Field(0, ge=0, le=4, description="Fenders locally painted (of 4)")
    bumper_changed: int = Field(0, ge=0, le=2, description="Bumpers replaced (of 2)")
    bumper_painted: int = Field(0, ge=0, le=2, description="Bumpers painted (of 2)")
    bumper_local: int = Field(0, ge=0, le=2, description="Bumpers locally painted (of 2)")

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

"""Pydantic v2 schemas for the NBA contract-value API.

The model consumes 36 derived features (per-36 stats, position dummies, role
flags, age polynomial). The API accepts raw per-game + advanced stats — the
shape a reader actually has — and runs the same `pipeline.features` transforms
that built the training table. No feature engineering decisions are re-invented
here; the API just routes inputs through the existing pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlayerStats(BaseModel):
    """One player-season's raw stats. Per-36 + dummies + flags are derived server-side."""

    model_config = ConfigDict(extra="forbid")

    # Identity (echoed back in the response)
    Player: str = Field(..., min_length=1, max_length=64, description="Player name; passthrough.")
    season: int = Field(..., ge=1950, le=2100, description="Season year (e.g. 2024 for the 2023-24 season).")

    # Bio / role
    Age: int = Field(..., ge=18, le=50)
    Pos: Literal["PG", "SG", "SF", "PF", "C", "PG-SG", "SG-SF", "SF-PF", "PF-C", "SG-PG", "SF-SG", "PF-SF", "C-PF"] = Field(
        ..., description="Primary position; multi-position labels keep first slot."
    )

    # Volume
    G: int = Field(..., ge=0, le=82)
    GS: int = Field(..., ge=0, le=82, description="Games started.")
    MP: float = Field(..., ge=0.0, le=48.0, description="Minutes per game.")

    # Per-game counting stats (per-36 versions computed server-side)
    PTS: float = Field(..., ge=0.0)
    TRB: float = Field(..., ge=0.0)
    AST: float = Field(..., ge=0.0)
    STL: float = Field(..., ge=0.0)
    BLK: float = Field(..., ge=0.0)
    TOV: float = Field(..., ge=0.0)
    FGA: float = Field(..., ge=0.0)
    ThreePA: float = Field(..., ge=0.0, alias="3PA", description="Three-point attempts per game.")
    FTA: float = Field(..., ge=0.0)

    # Shooting efficiency
    FGpct: float = Field(..., ge=0.0, le=1.0, alias="FG%")
    ThreePpct: float = Field(..., ge=0.0, le=1.0, alias="3P%")
    FTpct: float = Field(..., ge=0.0, le=1.0, alias="FT%")
    eFGpct: float = Field(..., ge=0.0, le=1.0, alias="eFG%")

    # Advanced
    PER: float = Field(..., description="Player Efficiency Rating (league avg ~15).")
    TSpct: float = Field(..., ge=0.0, le=1.0, alias="TS%")
    USGpct: float = Field(..., ge=0.0, le=100.0, alias="USG%")
    WS: float = Field(...)
    WSper48: float = Field(..., alias="WS/48")
    BPM: float = Field(...)
    OBPM: float = Field(...)
    DBPM: float = Field(...)
    VORP: float = Field(...)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    players: list[PlayerStats] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    Player: str
    season: int
    predicted_usd: int = Field(..., description="Point estimate in dollars.")
    predicted_m: float = Field(..., description="Same as predicted_usd, in millions, 2 dp.")
    tier: Literal["Max contract", "Star", "Starter", "Role player", "Bench / reserve"]
    cap_pct_2025: float = Field(..., description="% of 2025 salary cap ($140.6M).")
    interval_80_low_usd: int | None = Field(None, description="Empirical 80% interval (validation-season residual quantiles).")
    interval_80_high_usd: int | None = None
    stale: bool = Field(False, description="True when the requested season is more than one season past the newest training season.")


class FeatureDrift(BaseModel):
    psi: float
    status: str


class PredictionDrift(BaseModel):
    psi: float
    status: str
    median_predicted_usd: float
    reference_median_usd: float | None = None


class DriftResponse(BaseModel):
    n_rows: int
    reference: dict
    features: dict[str, FeatureDrift]
    n_shifted: int
    n_moderate: int
    worst: list = Field(default_factory=list)
    prediction_psi: PredictionDrift | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    drift: DriftResponse | None = Field(None, description="present for batches of 50+ rows")


class ModelInfo(BaseModel):
    """Honest model card. Notes the synthetic-data caveat on headline metrics."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    framework: str
    n_features: int
    feature_names: list[str]
    test_r2: float
    test_mae_usd: float
    test_mae_log: float
    best_params: dict
    honest_disclosure: str
    training_seasons: list[int] | None = None
    validation_season: int | None = None
    test_season: int | None = None
    trained_at: str | None = None
    data_source: str | None = None
    interval_80_coverage_test: float | None = None
    held_out_evaluation: dict | None = None
    error_by_salary_tier: list[dict] | None = None
    top_features: list[dict] | None = None
    stale_after_season: int | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    version: str = ""
    test_season: int | None = None
    drift_reference_loaded: bool = False
    uptime_s: float = 0.0

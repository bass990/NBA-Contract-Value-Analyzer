"""FastAPI service for the NBA contract-value model.

Reuses the project's own scoring and feature pipelines. The API does NOT
duplicate any modeling logic: it accepts the raw stats a reader actually has
(PTS, TRB, USG%, etc.), runs them through `pipeline.features` to derive
per-36 stats / position dummies / role flags / age polynomial, then calls
`model.score.predict_salary`. If the feature pipeline changes, this API
follows automatically.

Version 1.1 adds what the model bundle from scripts/train.py makes possible:
an empirical 80% prediction interval on every prediction, a `stale` flag when
the requested season is beyond what the model has seen, PSI drift monitoring
against the training feature distribution, held-out evaluation on the model
card, request ids and timing headers, and structured request logs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.monitoring import MIN_ROWS_FOR_DRIFT, drift_report, is_stale
from api.schemas import (
    BatchRequest,
    DriftResponse,
    HealthResponse,
    ModelInfo,
    PlayerStats,
    Prediction,
    PredictResponse,
)
from src.model.score import _align_features, load_model, predict_salary
from src.pipeline.features import (
    add_age_features,
    add_per_36,
    add_position_dummies,
    add_role_indicators,
)

MODEL_PATH = Path(os.getenv("NBA_MODEL_PATH", PROJECT_ROOT / "data" / "processed" / "model.pkl"))
EVAL_REPORT_PATH = PROJECT_ROOT / "artifacts" / "eval_report.json"
API_VERSION = "1.1.0"

# 2025-26 NBA salary cap, in dollars. Used to render tier + cap-% in responses.
SALARY_CAP_2025 = 140_600_000
POS_DUMMY_COLS = ["pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"]

log = logging.getLogger("nba_api")
logging.basicConfig(level=os.getenv("NBA_API_LOG_LEVEL", "INFO"), format="%(message)s")

_state: dict = {"model_loaded": False, "bundle": None, "eval_report": None, "started_at": None}


def _warmup_feature_frame(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{col: 0.0 for col in feature_names}])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at boot and warm the predict path."""
    bundle = load_model(MODEL_PATH)
    _ = predict_salary(bundle, _warmup_feature_frame(bundle["features"]))
    _state["bundle"] = bundle
    _state["eval_report"] = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8")) if EVAL_REPORT_PATH.exists() else None
    _state["model_loaded"] = True
    _state["started_at"] = time.time()
    log.info(json.dumps({"event": "startup", "features": len(bundle["features"]), "test_season": bundle.get("test_season"),
                         "intervals": "residual_quantiles_log" in bundle, "drift_reference": "reference" in bundle}))
    yield


app = FastAPI(
    title="NBA Contract Value API",
    version=API_VERSION,
    description=(
        "LightGBM regression on log(salary) predicting NBA player contract value from per-game + advanced stats, "
        "with an empirical 80% interval, a staleness flag and PSI drift monitoring. See /model-info for the model card "
        "and the synthetic-data caveat on headline metrics."
    ),
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("NBA_API_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


# The static website (website/) is served from the same origin when it is present (the Docker image
# bundles it), so the Docker port link opens the product and the Salary Estimator calls this API without CORS.
_WEBSITE_DIR = next((d for d in (Path(__file__).resolve().parent.parent / "website", Path("/app/website")) if (d / "index.html").exists()), None)


@app.get("/", include_in_schema=False)
def root():
    """The website when bundled, otherwise the interactive docs; never a bare 404."""
    if _WEBSITE_DIR is not None:
        return FileResponse(_WEBSITE_DIR / "index.html")
    return RedirectResponse(url="/docs")



@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    if request.url.path.startswith(("/predict", "/drift")):
        log.info(json.dumps({"event": "request", "id": rid, "path": request.url.path, "status": response.status_code, "ms": round(ms, 1)}))
    return response


def _require_loaded() -> dict:
    if not _state["model_loaded"]:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded yet")
    return _state["bundle"]


def _players_to_raw_frame(players: list[PlayerStats]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump(by_alias=True) for p in players])


def _ensure_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for col in POS_DUMMY_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(int)
    return df


def _engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_per_36(raw)
    df = add_position_dummies(df)
    df = add_role_indicators(df)
    df = add_age_features(df)
    return _ensure_position_dummies(df)


def _tier(salary_usd: float) -> str:
    if salary_usd >= 40_000_000:
        return "Max contract"
    if salary_usd >= 25_000_000:
        return "Star"
    if salary_usd >= 15_000_000:
        return "Starter"
    if salary_usd >= 7_000_000:
        return "Role player"
    return "Bench / reserve"


def _predict(players: list[PlayerStats]) -> tuple[pd.DataFrame, np.ndarray]:
    bundle = _require_loaded()
    try:
        feats = _engineer_features(_players_to_raw_frame(players))
        preds = predict_salary(bundle, feats)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if not np.all(np.isfinite(preds)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Model produced a non-finite prediction for this input")
    return feats, preds


def _build_predictions(players: list[PlayerStats], salaries_usd: np.ndarray) -> list[Prediction]:
    bundle = _state["bundle"]
    q = bundle.get("residual_quantiles_log")
    latest = max(bundle.get("training_seasons", [bundle.get("test_season", 2025)]) + [bundle.get("validation_season", 0)])
    out = []
    for player, sal in zip(players, salaries_usd, strict=True):
        sal_f = float(sal)
        lo = round(sal_f * float(np.exp(q["q10"]))) if q else None
        hi = round(sal_f * float(np.exp(q["q90"]))) if q else None
        out.append(Prediction(
            Player=player.Player, season=player.season, predicted_usd=round(sal_f), predicted_m=round(sal_f / 1_000_000, 2),
            tier=_tier(sal_f), cap_pct_2025=round(sal_f / SALARY_CAP_2025 * 100, 1),
            interval_80_low_usd=lo, interval_80_high_usd=hi, stale=is_stale(player.season, latest),
        ))
    return out


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    b = _state["bundle"] or {}
    return HealthResponse(status="ok" if _state["model_loaded"] else "loading", model_loaded=_state["model_loaded"], version=API_VERSION,
                          test_season=b.get("test_season"), drift_reference_loaded="reference" in b,
                          uptime_s=round(time.time() - _state["started_at"], 1) if _state["started_at"] else 0.0)


@app.get("/model-info", response_model=ModelInfo, tags=["meta"])
def model_info() -> ModelInfo:
    bundle = _require_loaded()
    rep = _state["eval_report"] or {}
    return ModelInfo(
        model_name="NBA Contract Value: LightGBM",
        model_version=API_VERSION,
        framework="lightgbm",
        n_features=len(bundle["features"]),
        feature_names=list(bundle["features"]),
        test_r2=float(bundle["test_r2"]),
        test_mae_usd=float(bundle["test_mae_usd"]),
        test_mae_log=float(bundle["test_mae_log"]),
        best_params={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in bundle["best_params"].items()},
        honest_disclosure=(
            f"Headline metrics (R² {bundle['test_r2']:.3f}, MAE ${bundle['test_mae_usd'] / 1e6:.2f}M) are from a synthetic-data run "
            "(seed 42, 1,400 player-seasons), the pipeline's reproducibility fallback: the salary sources (Spotrac, HoopsHype) sit "
            "behind Cloudflare and were not scraped. Real-data dev runs on a hand-curated ~75-row salary set land in R² 0.68-0.74. "
            "The 80% interval comes from validation-season residuals and covered "
            f"{100 * bundle.get('interval_coverage_80_test', float('nan')):.0f}% of test-season salaries; it is widest, and least reliable, "
            "for max contracts, which the synthetic generator caps at $60M."
        ),
        training_seasons=bundle.get("training_seasons"), validation_season=bundle.get("validation_season"),
        test_season=bundle.get("test_season"), trained_at=bundle.get("trained_at"), data_source=bundle.get("data_source"),
        interval_80_coverage_test=bundle.get("interval_coverage_80_test"),
        held_out_evaluation=rep.get("headline"), error_by_salary_tier=rep.get("by_salary_tier"), top_features=bundle.get("top_features"),
        stale_after_season=(max(bundle.get("training_seasons", [0]) + [bundle.get("validation_season", 0)]) + 1) if bundle.get("training_seasons") else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(player: PlayerStats) -> PredictResponse:
    """Score one player-season: salary point estimate, 80% interval, tier, % of cap, staleness."""
    _, preds = _predict([player])
    return PredictResponse(predictions=_build_predictions([player], preds))


@app.post("/predict-batch", response_model=PredictResponse, tags=["predict"])
def predict_batch(req: BatchRequest) -> PredictResponse:
    """Score 1..1,000 player-seasons; order preserved. Batches of 50+ rows get a drift summary."""
    feats, preds = _predict(req.players)
    bundle = _state["bundle"]
    drift = None
    if "reference" in bundle and len(req.players) >= MIN_ROWS_FOR_DRIFT:
        drift = drift_report(_align_features(bundle, feats), bundle["reference"], preds)
    return PredictResponse(predictions=_build_predictions(req.players, preds), drift=drift)


@app.post("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift(req: BatchRequest) -> DriftResponse:
    """PSI of a batch's model features (and predicted log-salary) against the training distribution."""
    bundle = _require_loaded()
    if "reference" not in bundle:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model bundle has no drift reference; retrain with scripts/train.py")
    if len(req.players) < MIN_ROWS_FOR_DRIFT:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Drift needs at least {MIN_ROWS_FOR_DRIFT} rows; got {len(req.players)}")
    feats, preds = _predict(req.players)
    return DriftResponse(**drift_report(_align_features(bundle, feats), bundle["reference"], preds))


# Registered last on purpose: Starlette matches in registration order, so every API route above wins and only
# the website's own files (style.css, app.js) fall through to the static mount.
if _WEBSITE_DIR is not None:
    app.mount("/", StaticFiles(directory=str(_WEBSITE_DIR)), name="site")

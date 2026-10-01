"""Input-drift and staleness monitoring for the salary API.

Population Stability Index (PSI) per model feature against the reference
histograms frozen at training time (stored inside the model bundle by
scripts/train.py). PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 shifted.

Staleness is the other failure mode for this model: the salary cap grows
every season, so a model whose newest training season is more than one
season behind the requested season is extrapolating on cap inflation it has
never seen. /model-info and every prediction carry a `stale` flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ROWS_FOR_DRIFT = 50
_EPS = 1e-6
STABLE, MODERATE = 0.10, 0.25


def _hist_share(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total


def psi(expected_share: np.ndarray, actual_share: np.ndarray) -> float:
    e = np.clip(expected_share, _EPS, None)
    a = np.clip(actual_share, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float) -> str:
    if value < STABLE:
        return "stable"
    if value < MODERATE:
        return "moderate"
    return "shifted"


def drift_report(features: pd.DataFrame, reference: dict, predictions_usd: np.ndarray | None = None) -> dict:
    per_feature: dict[str, dict] = {}
    for name, ref in reference["features"].items():
        if name not in features.columns:
            continue
        col = pd.to_numeric(features[name], errors="coerce").dropna().to_numpy(dtype=float)
        if col.size == 0:
            continue
        value = psi(np.asarray(ref["share"]), _hist_share(col, ref["edges"]))
        per_feature[name] = {"psi": round(value, 4), "status": _status(value)}
    out = {
        "n_rows": len(features),
        "reference": {"source": reference.get("source"), "n_rows": reference.get("n_rows"), "seasons": reference.get("seasons")},
        "features": per_feature,
        "n_shifted": sum(1 for v in per_feature.values() if v["status"] == "shifted"),
        "n_moderate": sum(1 for v in per_feature.values() if v["status"] == "moderate"),
        "worst": sorted(per_feature.items(), key=lambda kv: -kv[1]["psi"])[:5],
    }
    if predictions_usd is not None and "prediction" in reference:
        logp = np.log(np.clip(np.asarray(predictions_usd, dtype=float), 1, None))
        value = psi(np.asarray(reference["prediction"]["share"]), _hist_share(logp, reference["prediction"]["edges"]))
        out["prediction_psi"] = {"psi": round(value, 4), "status": _status(value),
                                 "median_predicted_usd": float(np.median(predictions_usd)),
                                 "reference_median_usd": reference["prediction"].get("median_usd")}
    return out


def is_stale(requested_season: int, latest_training_season: int, tolerance: int = 1) -> bool:
    """True when the request is more than `tolerance` seasons past the newest training season."""
    return requested_season - latest_training_season > tolerance

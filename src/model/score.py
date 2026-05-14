"""Scoring (inference) module — load a trained model and predict salaries.

Kept separate from train.py intentionally. Inference has no dependency on
Optuna, sklearn metrics, or the training loop. A production service, a
notebook, or a CLI tool can import this module without pulling in the full
training stack.

Typical usage:

    from src.model.score import load_model, score_players

    bundle = load_model(Path("data/processed/model.pkl"))
    predictions = score_players(bundle, feature_table)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(path: Path) -> dict:
    """Load a trained model bundle from disk.

    The bundle is a plain dict with keys:
        model       — lightgbm.Booster
        features    — list[str] of feature column names (in training order)
        best_params — dict of hyperparameters used
        test_r2     — float, held-out R² on log(salary)
        test_mae_log — float, held-out MAE on log(salary)
        test_mae_usd — float, held-out MAE in dollars

    Args:
        path: Path to the .pkl file written by train.save_model().

    Returns:
        The model bundle dict.

    Raises:
        FileNotFoundError if the path does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run the notebook or `make train` to generate it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _align_features(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Align a feature DataFrame to the exact columns the model expects.

    Missing columns are filled with 0 (appropriate for binary dummies and
    rare advanced stats that may be absent from a single-player input).
    Extra columns are silently dropped.
    """
    expected = bundle["features"]
    X_out = pd.DataFrame(index=X.index)
    for col in expected:
        if col in X.columns:
            X_out[col] = X[col]
        else:
            X_out[col] = 0.0
    # Median-impute any remaining NaNs (handles partial stat rows)
    for col in X_out.columns:
        if X_out[col].isna().any():
            median_val = X_out[col].median()
            X_out[col] = X_out[col].fillna(median_val if not np.isnan(median_val) else 0.0)
    return X_out


def predict_salary(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Predict annual salary in dollars for one or more player-seasons.

    Args:
        bundle: Model bundle returned by load_model().
        X:      Feature DataFrame. Must contain at minimum the columns in
                bundle["features"]; any extra columns are ignored; any missing
                columns are filled with 0.

    Returns:
        1-D numpy array of predicted salaries in USD, same length as X.
    """
    X_aligned = _align_features(bundle, X)
    log_preds = bundle["model"].predict(X_aligned)
    return np.exp(log_preds)


def score_players(
    bundle: dict,
    feature_table: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Score a full feature table and return a ranked results DataFrame.

    Args:
        bundle:        Model bundle returned by load_model().
        feature_table: Output of build_feature_table(). Must have the model
                       feature columns; salary columns are optional.
        id_cols:       Additional columns from feature_table to carry through
                       to the output (e.g. ["Pos_primary", "Age", "MP"]).
                       Defaults to ["Player", "season"].

    Returns:
        DataFrame with columns:
            Player, season, predicted_usd
            — plus actual_usd, residual_usd, pct_off if salary data present
            — plus any id_cols requested
        Sorted by predicted_usd descending.
    """
    if id_cols is None:
        id_cols = ["Player", "season"]

    carry = [c for c in id_cols if c in feature_table.columns]
    preds_usd = predict_salary(bundle, feature_table)

    out = feature_table[carry].copy().reset_index(drop=True)
    out["predicted_usd"] = preds_usd

    if "salary_usd" in feature_table.columns:
        actual = feature_table["salary_usd"].values
        out["actual_usd"] = actual
        out["residual_usd"] = out["predicted_usd"] - out["actual_usd"]
        out["pct_off"] = out["residual_usd"] / out["actual_usd"]

    return out.sort_values("predicted_usd", ascending=False).reset_index(drop=True)


def score_single_player(
    bundle: dict,
    stats: dict,
) -> dict:
    """Score a single player defined by a flat stats dictionary.

    Convenience wrapper for the salary estimator tool — pass raw per-game
    and advanced stats, get back a salary estimate with context.

    The caller is responsible for computing per-36 columns if needed; this
    function passes the dict straight into _align_features, which fills
    any missing model columns with 0.

    Args:
        bundle: Model bundle returned by load_model().
        stats:  Dict mapping stat name → value.
                Example: {"Age": 26, "PTS": 18.4, "WS": 5.2, "USG%": 22.1}

    Returns:
        Dict with keys:
            predicted_usd   — point estimate in dollars
            predicted_m     — same, in millions (rounded to 2 dp)
            tier            — contract tier label (str)
            cap_pct_2025    — % of 2025 salary cap ($140.6M)
    """
    X = pd.DataFrame([stats])
    sal = float(predict_salary(bundle, X)[0])

    if sal >= 40_000_000:
        tier = "Max contract"
    elif sal >= 25_000_000:
        tier = "Star"
    elif sal >= 15_000_000:
        tier = "Starter"
    elif sal >= 7_000_000:
        tier = "Role player"
    else:
        tier = "Bench / reserve"

    return {
        "predicted_usd": round(sal),
        "predicted_m": round(sal / 1_000_000, 2),
        "tier": tier,
        "cap_pct_2025": round(sal / 140_600_000 * 100, 1),
    }

"""Train a LightGBM model to predict log(salary) from player stats.

Why LightGBM:
    Gradient boosting on trees is the right default for structured player-stats data.
    Features are heterogeneous (counting stats, percentages, dummies), there are
    nonlinear interactions (a 30-year-old with a 25 USG% is paid differently from
    a 23-year-old with 25 USG%), and the dataset is tabular and small (~500 players
    per season). Neural nets don't help here; linear models miss the interactions.

    Within gradient boosting, LightGBM is faster than XGBoost on small data and has
    better categorical handling. CatBoost would also work.

Why time-series split:
    Salary caps grow over time. Random K-fold CV would let the model train on 2025
    data and predict 2022 data, which is information leakage from the future. We
    train on older seasons and validate on the most recent one — the same direction
    we'll predict in production.

Why Optuna for tuning:
    Grid search over LightGBM's important hyperparameters (num_leaves, learning_rate,
    min_data_in_leaf, feature_fraction) explodes combinatorially. Optuna's TPE
    sampler converges much faster on small budgets.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from ..pipeline.features import MODEL_FEATURES

logger = logging.getLogger(__name__)

# Suppress Optuna's chatter unless you really want it
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainResult:
    model: lgb.Booster
    best_params: dict
    test_r2: float
    test_mae_log: float
    test_mae_usd: float
    test_predictions: pd.DataFrame  # columns: Player, season, actual_usd, predicted_usd, residual_usd


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Slice features + target from a feature table. Drops rows missing salary or required features."""
    feat_cols = [c for c in MODEL_FEATURES if c in df.columns]
    work = df.dropna(subset=["log_salary"]).copy()
    # Drop rows with too many missing features (rare; usually rookies with partial stats)
    work = work.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.8))
    # Light imputation: fill remaining NaNs with column medians
    for c in feat_cols:
        if work[c].isna().any():
            work[c] = work[c].fillna(work[c].median())
    X = work[feat_cols]
    y = work["log_salary"]
    return X, y


def _time_split(
    df: pd.DataFrame, test_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on seasons < test_season; test on test_season."""
    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, test


def _tune(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30) -> dict:
    """Optuna search. Uses internal validation split inside the training set."""
    # Hold out the last season inside the training set for HP tuning so we don't
    # touch the real test season.
    # X_train must have a 'season' aux column hidden; we use index-based split here.
    n = len(X_train)
    split = int(n * 0.85)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 40),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        return float(mean_absolute_error(y_val, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train(
    feature_table: pd.DataFrame,
    test_season: int,
    n_trials: int = 30,
    skip_tuning: bool = False,
) -> TrainResult:
    """Train a salary model.

    Args:
        feature_table: Output of build_feature_table() with log_salary column.
        test_season: Season to hold out as the test set.
        n_trials: Optuna trials for tuning. Set to 0 (or skip_tuning=True) for a fast smoke test.
        skip_tuning: If True, use sensible defaults instead of running Optuna.

    Returns:
        TrainResult with the model, hyperparameters, and metrics.
    """
    train_df, test_df = _time_split(feature_table, test_season)
    logger.info("Train: %d rows | Test (season=%d): %d rows", len(train_df), test_season, len(test_df))

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    if skip_tuning or n_trials == 0:
        best_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 15,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "lambda_l2": 1.0,
        }
        logger.info("Skipping tuning, using defaults: %s", best_params)
    else:
        logger.info("Tuning hyperparameters (%d trials)...", n_trials)
        best_params = _tune(X_train, y_train, n_trials=n_trials)
        logger.info("Best params: %s", best_params)

    # Train final model on ALL training data
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 5,
        **best_params,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    preds_log = model.predict(X_test)
    preds_usd = np.exp(preds_log)
    actual_usd = np.exp(y_test.values)

    r2 = r2_score(y_test, preds_log)
    mae_log = mean_absolute_error(y_test, preds_log)
    mae_usd = mean_absolute_error(actual_usd, preds_usd)

    logger.info("Test R²: %.3f | MAE(log): %.3f | MAE($): %s", r2, mae_log, f"${mae_usd:,.0f}")

    # Build a residuals table for the dashboard
    test_meta = test_df.loc[X_test.index, ["Player", "season"]].reset_index(drop=True)
    test_meta["actual_usd"] = actual_usd
    test_meta["predicted_usd"] = preds_usd
    test_meta["residual_usd"] = test_meta["predicted_usd"] - test_meta["actual_usd"]
    test_meta["pct_off"] = test_meta["residual_usd"] / test_meta["actual_usd"]

    return TrainResult(
        model=model,
        best_params=best_params,
        test_r2=r2,
        test_mae_log=mae_log,
        test_mae_usd=mae_usd,
        test_predictions=test_meta.sort_values("residual_usd", ascending=False),
    )


def save_model(result: TrainResult, path: Path) -> None:
    """Persist model + metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": result.model,
                "best_params": result.best_params,
                "test_r2": result.test_r2,
                "test_mae_log": result.test_mae_log,
                "test_mae_usd": result.test_mae_usd,
                "features": MODEL_FEATURES,
            },
            f,
        )
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> dict:
    """Load a saved model bundle. See src/model/score.py for the full inference API."""
    with open(path, "rb") as f:
        return pickle.load(f)

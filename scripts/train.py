"""Deterministic training run: synthetic data (seed 42) -> model bundle + evaluation report.

    python scripts/train.py [--test-season 2025] [--out data/processed/model.pkl]

What changed versus the notebook's original run
------------------------------------------------
* Early stopping used to watch the TEST season (`valid_sets=[dtest]`), which
  leaks the held-out year into the choice of tree count. Here the last
  training season (2024) is the early-stopping set and 2025 is touched only
  once, at the end, to report metrics. The headline moves accordingly and the
  README says so.
* The bundle now carries what the API needs to be honest at request time:
  residual quantiles from the validation season (an empirical 80% prediction
  interval), reference histograms of every feature and of the predicted
  log-salary (drift), feature importance, the seasons trained on and the
  training timestamp.
* `artifacts/eval_report.json` records held-out metrics, error by salary tier
  and by position, and the realised coverage of the 80% interval on the test
  season.

Every number is reproducible: `make train` twice gives byte-identical metrics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.train import _prepare_xy  # noqa: E402
from src.pipeline.features import MODEL_FEATURES, build_feature_table  # noqa: E402
from src.pipeline.synthetic import generate_synthetic_dataset  # noqa: E402

DEFAULT_PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 15, "feature_fraction": 0.85,
                  "bagging_fraction": 0.85, "lambda_l2": 1.0}
N_BINS = 10
TIERS = [(0, 7e6, "Bench / reserve"), (7e6, 15e6, "Role player"), (15e6, 25e6, "Starter"), (25e6, 40e6, "Star"), (40e6, np.inf, "Max contract")]


def _edges(values: np.ndarray) -> list[float]:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return [float(e) for e in edges]


def _reference(X: pd.DataFrame, log_preds: np.ndarray, seasons: list[int]) -> dict:
    feats = {}
    for c in X.columns:
        col = pd.to_numeric(X[c], errors="coerce").dropna().to_numpy(dtype=float)
        if col.size == 0:
            continue
        edges = _edges(col)
        counts, _ = np.histogram(col, bins=np.asarray(edges))
        feats[c] = {"edges": edges, "share": [float(v) / counts.sum() for v in counts]}
    p_edges = _edges(log_preds)
    p_counts, _ = np.histogram(log_preds, bins=np.asarray(p_edges))
    return {"source": "synthetic seed 42, training seasons", "n_rows": int(len(X)), "seasons": seasons, "features": feats,
            "prediction": {"edges": p_edges, "share": [float(v) / p_counts.sum() for v in p_counts],
                           "median_usd": float(np.exp(np.median(log_preds)))}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "model.pkl"))
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "eval_report.json"))
    args = ap.parse_args()

    pg, adv, sal = generate_synthetic_dataset(seed=args.seed)
    ft = build_feature_table(pg, adv, sal)
    seasons = sorted(int(s) for s in ft["season"].unique())
    val_season = max(s for s in seasons if s < args.test_season)
    train_df = ft[ft["season"] < val_season]
    val_df = ft[ft["season"] == val_season]
    test_df = ft[ft["season"] == args.test_season]
    X_tr, y_tr = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)
    X_te, y_te = _prepare_xy(test_df)
    print(f"train {sorted(train_df.season.unique())}: {len(X_tr)} rows | early-stop on {val_season}: {len(X_val)} | test {args.test_season}: {len(X_te)}")

    params = {"objective": "regression", "metric": "mae", "verbosity": -1, "boosting_type": "gbdt", "bagging_freq": 5,
              "seed": args.seed, "deterministic": True, "num_threads": 1, **DEFAULT_PARAMS}
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # validation residuals -> empirical prediction interval (log scale, symmetric use)
    val_resid = y_val.to_numpy() - model.predict(X_val)
    q10, q50, q90 = (float(np.quantile(val_resid, q)) for q in (0.10, 0.50, 0.90))

    log_pred = model.predict(X_te)
    pred_usd, actual_usd = np.exp(log_pred), np.exp(y_te.to_numpy())
    lo, hi = np.exp(log_pred + q10), np.exp(log_pred + q90)
    coverage = float(np.mean((actual_usd >= lo) & (actual_usd <= hi)))
    r2, mae_log, mae_usd = float(r2_score(y_te, log_pred)), float(mean_absolute_error(y_te, log_pred)), float(mean_absolute_error(actual_usd, pred_usd))
    mape = float(np.mean(np.abs(pred_usd - actual_usd) / actual_usd))

    by_tier = []
    for lo_t, hi_t, name in TIERS:
        m = (actual_usd >= lo_t) & (actual_usd < hi_t)
        if m.sum() == 0:
            continue
        by_tier.append({"tier": name, "n": int(m.sum()), "mae_usd": round(float(np.mean(np.abs(pred_usd[m] - actual_usd[m])))),
                        "mape": round(float(np.mean(np.abs(pred_usd[m] - actual_usd[m]) / actual_usd[m])), 3),
                        "bias_usd": round(float(np.mean(pred_usd[m] - actual_usd[m]))), "coverage_80": round(float(np.mean((actual_usd[m] >= lo[m]) & (actual_usd[m] <= hi[m]))), 3)})
    pos_col = test_df.loc[X_te.index, "Pos_primary"].to_numpy()
    by_pos = []
    for p in sorted(set(pos_col)):
        m = pos_col == p
        by_pos.append({"position": str(p), "n": int(m.sum()), "mae_usd": round(float(np.mean(np.abs(pred_usd[m] - actual_usd[m])))),
                       "bias_usd": round(float(np.mean(pred_usd[m] - actual_usd[m])))})
    importance = sorted(zip(X_tr.columns, model.feature_importance(importance_type="gain"), strict=True), key=lambda kv: -kv[1])
    top_features = [{"feature": f, "gain_share": round(float(g) / float(sum(v for _, v in importance)), 4)} for f, g in importance[:10]]

    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "model": model, "best_params": DEFAULT_PARAMS, "features": MODEL_FEATURES,
        "test_r2": r2, "test_mae_log": mae_log, "test_mae_usd": mae_usd,
        "test_season": args.test_season, "validation_season": val_season, "training_seasons": sorted(int(s) for s in train_df.season.unique()),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "trained_at": trained_at, "seed": args.seed,
        "data_source": "synthetic (src/pipeline/synthetic.py, seed 42); see README honest disclosure",
        "residual_quantiles_log": {"q10": q10, "q50": q50, "q90": q90},
        "interval_coverage_80_test": coverage,
        "reference": _reference(X_tr, model.predict(X_tr), sorted(int(s) for s in train_df.season.unique())),
        "top_features": top_features,
        "lightgbm_version": lgb.__version__,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    report = {"trained_at": trained_at, "data": bundle["data_source"], "training_seasons": bundle["training_seasons"],
              "validation_season": val_season, "test_season": args.test_season, "n_train": bundle["n_train"], "n_test": bundle["n_test"],
              "best_iteration": int(model.best_iteration), "params": DEFAULT_PARAMS,
              "headline": {"r2_log": round(r2, 4), "mae_log": round(mae_log, 4), "mae_usd": round(mae_usd), "mape": round(mape, 4),
                           "interval_80_coverage": round(coverage, 4), "interval_80_half_width_log": round((q90 - q10) / 2, 4)},
              "by_salary_tier": by_tier, "by_position": by_pos, "top_features": top_features,
              "note": "early stopping now uses the validation season, not the test season (the notebook run used the test season)"}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["headline"], indent=1))
    print("by tier:", [(t["tier"], t["mae_usd"], t["coverage_80"]) for t in by_tier])
    print(f"best iteration {model.best_iteration}; wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

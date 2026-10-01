"""Regenerate data/processed/predictions_latest.csv from the committed model bundle.

    python scripts/predict_latest.py [--test-season 2025] [--seed 42]

Scores the held-out season of the synthetic dataset (same seed as scripts/train.py)
with data/processed/model.pkl, so the file the dashboard and the Space ship always
matches the bundle's headline numbers. Columns are the ones the dashboard expects:
Player, season, actual_usd, predicted_usd, residual_usd, pct_off, Pos_primary, Age, MP.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.score import score_players
from src.pipeline.features import build_feature_table
from src.pipeline.synthetic import generate_synthetic_dataset

COLUMNS = ["Player", "season", "actual_usd", "predicted_usd", "residual_usd", "pct_off", "Pos_primary", "Age", "MP"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=None, help="defaults to the bundle's test season")
    ap.add_argument("--seed", type=int, default=None, help="defaults to the bundle's seed")
    ap.add_argument("--bundle", default=str(ROOT / "data" / "processed" / "model.pkl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "predictions_latest.csv"))
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    season = args.test_season or int(bundle["test_season"])
    seed = args.seed if args.seed is not None else int(bundle.get("seed", 42))

    pg, adv, sal = generate_synthetic_dataset(seed=seed)
    ft = build_feature_table(pg, adv, sal)
    test = ft[ft["season"] == season].dropna(subset=["salary_usd"])
    out = score_players(bundle, test, id_cols=["Player", "season", "Pos_primary", "Age", "MP"])
    out = out[COLUMNS]
    out.to_csv(args.out, index=False)

    r2 = r2_score(np.log(out["actual_usd"]), np.log(out["predicted_usd"]))
    mae = float(np.mean(np.abs(out["predicted_usd"] - out["actual_usd"])))
    print(f"{len(out)} rows for season {season} (seed {seed}) -> {args.out}")
    print(f"log-R2 {r2:.4f} vs bundle {bundle['test_r2']:.4f}; MAE ${mae:,.0f} vs bundle ${bundle['test_mae_usd']:,.0f}")
    if abs(r2 - bundle["test_r2"]) > 1e-6:
        print("WARNING: the scored season does not reproduce the bundle's headline; check seed/season")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

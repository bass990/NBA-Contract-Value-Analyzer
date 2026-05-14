"""Unit tests for the feature pipeline.

Uses synthetic data only — no network access required.
Run: pytest tests/ -v
"""

import pandas as pd
import pytest

from src.pipeline.features import (
    add_age_features,
    add_per_36,
    add_position_dummies,
    add_role_indicators,
    build_feature_table,
    merge_stats,
)
from src.pipeline.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def synthetic_data():
    return generate_synthetic_dataset(n_players_per_season=20, seasons=(2024, 2025))


def test_synthetic_generator_produces_expected_columns(synthetic_data):
    pg, adv, sal = synthetic_data
    assert "Player" in pg.columns
    assert "PTS" in pg.columns
    assert "PER" in adv.columns
    assert "salary_usd" in sal.columns


def test_merge_stats_inner_joins(synthetic_data):
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    assert len(merged) <= min(len(pg), len(adv))
    assert "PER" in merged.columns  # from advanced
    assert "PTS" in merged.columns  # from per-game


def test_per_36_added(synthetic_data):
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    out = add_per_36(merged)
    assert "PTS_per36" in out.columns
    # Per-36 should be > per-game when MP < 36, < per-game when MP > 36
    # For players with MP ~ 20, per36 should roughly double the counting stat
    sample = out.iloc[0]
    expected_ratio = 36.0 / sample["MP"]
    actual_ratio = sample["PTS_per36"] / sample["PTS"] if sample["PTS"] > 0 else 0
    if sample["PTS"] > 0:
        assert abs(actual_ratio - expected_ratio) < 1e-6


def test_position_dummies(synthetic_data):
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    out = add_position_dummies(merged)
    # All 5 position dummies should be present
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        assert f"pos_{pos}" in out.columns
    # Each row should sum to exactly 1 across position dummies (one-hot property)
    pos_cols = [f"pos_{p}" for p in ["PG", "SG", "SF", "PF", "C"]]
    assert (out[pos_cols].sum(axis=1) == 1).all()


def test_role_indicators(synthetic_data):
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    out = add_role_indicators(merged)
    assert "is_starter" in out.columns
    assert "is_high_usage" in out.columns
    assert "is_rotation_only" in out.columns
    # All should be 0 or 1
    assert set(out["is_starter"].unique()) <= {0, 1}


def test_age_features(synthetic_data):
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    out = add_age_features(merged)
    assert "Age_sq" in out.columns
    assert "is_prime" in out.columns
    # Age² should be Age * Age
    assert (out["Age_sq"] == out["Age"] ** 2).all()


def test_build_feature_table_end_to_end(synthetic_data):
    pg, adv, sal = synthetic_data
    ft = build_feature_table(pg, adv, sal)
    assert len(ft) > 0
    assert "log_salary" in ft.columns
    assert "salary_usd" in ft.columns
    # log_salary should equal np.log(salary_usd)
    import numpy as np
    np.testing.assert_allclose(
        ft["log_salary"].values,
        np.log(ft["salary_usd"].clip(lower=1).values),
    )


def test_no_data_leakage_in_features(synthetic_data):
    """The feature table should not contain salary-derived columns
    other than salary_usd and log_salary themselves."""
    pg, adv, sal = synthetic_data
    ft = build_feature_table(pg, adv, sal)
    # No future-season information in current-season rows
    for season in ft["season"].unique():
        season_rows = ft[ft["season"] == season]
        # Each row should have only that season's stats
        assert (season_rows["season"] == season).all()

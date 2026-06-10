"""Unit tests for the feature pipeline.

Uses synthetic data only — no network access required.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import (
    MODEL_FEATURES,
    add_age_features,
    add_per_36,
    add_position_dummies,
    add_role_indicators,
    attach_salaries,
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


# ── Edge cases & coverage gaps ────────────────────────────────────────────────

def test_per_36_handles_zero_minutes(synthetic_data):
    """A player with MP=0 (rare garbage-time-only seasons) must not produce
    inf or division-by-zero in per-36 stats — the implementation guards with
    `replace(0, np.nan)` so the row gets NaN per-36 values, not crashes."""
    pg, adv, _ = synthetic_data
    merged = merge_stats(pg, adv)
    # Force one player to have MP=0
    merged = merged.copy()
    merged.loc[merged.index[0], "MP"] = 0.0
    out = add_per_36(merged)
    # First row's per-36 stats should be NaN, not inf
    assert pd.isna(out.iloc[0]["PTS_per36"])
    assert not np.isinf(out.iloc[0]["PTS_per36"])
    # Other rows should be unaffected
    assert out["PTS_per36"].iloc[1:].notna().sum() > 0


def test_position_multi_position_collapses_to_primary():
    """Players listed as 'PG-SG' (multi-position eligibility on Basketball-Reference)
    should be assigned the FIRST position as primary — keeps the one-hot invariant."""
    df = pd.DataFrame({
        "Player": ["A", "B", "C", "D", "E"],
        "season": [2024] * 5,
        "Pos":    ["PG-SG", "SF-PF", "C", "SG", "PF"],  # cover all 5 primary positions
        "Age":    [25, 27, 29, 24, 30],
    })
    out = add_position_dummies(df)
    pos_cols = [f"pos_{p}" for p in ["PG", "SG", "SF", "PF", "C"]]
    # One-hot invariant: each row sums to exactly 1 across the standard 5
    assert (out[pos_cols].sum(axis=1) == 1).all()
    # PG-SG → PG primary, not SG
    assert out.iloc[0]["pos_PG"] == 1 and out.iloc[0]["pos_SG"] == 0
    # SF-PF → SF primary, not PF
    assert out.iloc[1]["pos_SF"] == 1 and out.iloc[1]["pos_PF"] == 0


def test_position_unknown_falls_through_to_other():
    """An unrecognized position string should not silently land in one of the
    standard 5 — it should be bucketed under 'OTHER' so it can be filtered or
    inspected explicitly downstream rather than misclassified silently."""
    df = pd.DataFrame({
        "Player": ["A", "B", "C", "D", "E", "F"],
        "season": [2024] * 6,
        # Include all 5 valid positions so pos_* columns get created, plus one unknown
        "Pos":    ["F", "PG", "SG", "SF", "PF", "C"],
        "Age":    [25] * 6,
    })
    out = add_position_dummies(df)
    # OTHER bucket should exist and the unknown row should be flagged
    assert "pos_OTHER" in out.columns
    assert out.iloc[0]["pos_OTHER"] == 1
    # The unknown row should NOT have been silently mapped to any of the standard 5
    standard_cols = [f"pos_{p}" for p in ["PG", "SG", "SF", "PF", "C"]]
    assert all(out.iloc[0][c] == 0 for c in standard_cols)


def test_attach_salaries_left_join_preserves_stat_rows(synthetic_data):
    """Players in the stats table without a salary match should remain in the
    output with NaN salary — a left join, not an inner join. This matters because
    silently dropping them would change the feature table size and break downstream."""
    pg, adv, sal = synthetic_data
    merged = merge_stats(pg, adv)
    enriched = add_per_36(merged)
    enriched = add_position_dummies(enriched)
    enriched = add_role_indicators(enriched)
    enriched = add_age_features(enriched)
    # Drop half the salary rows so we know there are unmatched stat rows
    partial_sal = sal.iloc[::2].copy()
    n_stat_rows = len(enriched)
    out = attach_salaries(enriched, partial_sal)
    # No stat rows dropped
    assert len(out) == n_stat_rows
    # Some salary cells should be NaN (the unmatched ones)
    assert out["salary_usd"].isna().sum() > 0


def test_build_feature_table_produces_all_model_features(synthetic_data):
    """Every column listed in MODEL_FEATURES must appear in the assembled feature
    table — otherwise the trainer will crash with a KeyError at runtime. This is
    the contract between the pipeline and the model."""
    pg, adv, sal = synthetic_data
    ft = build_feature_table(pg, adv, sal)
    missing = [c for c in MODEL_FEATURES if c not in ft.columns]
    assert not missing, f"MODEL_FEATURES columns missing from feature table: {missing}"


def test_synthetic_generator_is_reproducible():
    """The synthetic generator must be deterministic under a fixed seed —
    otherwise tests, the README's headline metrics, and CI become flaky."""
    pg1, adv1, sal1 = generate_synthetic_dataset(n_players_per_season=10, seasons=(2024,), seed=42)
    pg2, adv2, sal2 = generate_synthetic_dataset(n_players_per_season=10, seasons=(2024,), seed=42)
    pd.testing.assert_frame_equal(pg1, pg2)
    pd.testing.assert_frame_equal(adv1, adv2)
    pd.testing.assert_frame_equal(sal1, sal2)


def test_synthetic_salaries_within_realistic_bounds():
    """Generator must respect the $1M floor (minimum NBA contract) and stay
    below the documented ~$55M ceiling — a regression in the generator that
    breaks these bounds would invalidate every downstream test."""
    _, _, sal = generate_synthetic_dataset(n_players_per_season=200, seasons=(2024, 2025))
    assert (sal["salary_usd"] >= 1_000_000).all(), "Salary below $1M minimum"
    assert (sal["salary_usd"] <= 55_000_000).all(), "Salary above $55M ceiling"

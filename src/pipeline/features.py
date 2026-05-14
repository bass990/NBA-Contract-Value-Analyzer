"""Feature engineering: turn scraped per-game and advanced stats into a model-ready table.

Design choices made here that an interviewer might ask about:

1. WHY PER-36 STATS instead of per-game?
   A bench player who averages 22 minutes and a starter who averages 36 minutes are
   playing different volumes of basketball. Per-game stats reward minutes, which the
   salary model already partially captures via minutes-played features. Per-36 stats
   normalize for playing time, so the model can distinguish "efficient when on the
   floor" from "on the floor a lot." Both signals matter; isolating them helps.

2. WHY LOG-SALARY as the target?
   NBA salaries span $1M (min contract) to $55M+ (max). That's two orders of magnitude.
   A linear model loss in dollars would optimize to predict the median superstar well
   and ignore the bottom half entirely. log(salary) makes the loss surface symmetric
   across the salary range — a 20% miss on a $40M player and a 20% miss on a $2M
   player contribute equally. We exponentiate predictions back for display.

3. WHY position dummies?
   Centers and guards are paid by different supply curves. There are ~30 starting
   centers in the league; there are ~120 guards. Position-specific features let the
   model learn position-conditional pay curves.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns we'll keep from per-game stats (the ones that survive feature engineering)
PER_GAME_KEEP = [
    "Player", "season", "Pos", "Age", "Tm", "G", "GS", "MP",
    "FG", "FGA", "3P", "3PA", "FT", "FTA",
    "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "FG%", "3P%", "FT%", "eFG%",
]

# Advanced stats we want
ADVANCED_KEEP = [
    "Player", "season", "PER", "TS%", "USG%", "WS", "WS/48",
    "BPM", "OBPM", "DBPM", "VORP",
]


def merge_stats(per_game: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-game and advanced stats on (Player, season).

    We use an inner join intentionally: if a player is missing from one table,
    they're almost certainly a non-rotation player with too few minutes to model.
    """
    pg = per_game[[c for c in PER_GAME_KEEP if c in per_game.columns]].copy()
    adv = advanced[[c for c in ADVANCED_KEEP if c in advanced.columns]].copy()

    # Drop duplicates that may exist if scraper hiccupped
    pg = pg.drop_duplicates(subset=["Player", "season"])
    adv = adv.drop_duplicates(subset=["Player", "season"])

    merged = pg.merge(adv, on=["Player", "season"], how="inner")
    logger.info(
        "Merged stats: %d players (pg=%d, adv=%d)", len(merged), len(pg), len(adv)
    )
    return merged


def add_per_36(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-36-minute versions of counting stats.

    Per-36 = (stat / total_minutes) * 36, where total_minutes = MP_per_game * games_played
    But since MP is already per-game, per-36 = stat_per_game * 36 / MP_per_game
    """
    df = df.copy()
    counting_stats = ["PTS", "TRB", "AST", "STL", "BLK", "TOV", "FGA", "3PA", "FTA"]

    # Guard against MP=0 (rare but possible if a player only played end-of-game garbage time)
    safe_mp = df["MP"].replace(0, np.nan)

    for stat in counting_stats:
        if stat in df.columns:
            df[f"{stat}_per36"] = df[stat] * 36.0 / safe_mp

    return df


def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode position. Some players have multi-position labels like 'PG-SG';
    we take the first one as primary.
    """
    df = df.copy()
    df["Pos_primary"] = df["Pos"].astype(str).str.split("-").str[0]

    # Standard 5 positions; anything unexpected becomes "OTHER"
    valid = {"PG", "SG", "SF", "PF", "C"}
    df["Pos_primary"] = df["Pos_primary"].where(df["Pos_primary"].isin(valid), "OTHER")

    dummies = pd.get_dummies(df["Pos_primary"], prefix="pos")
    df = pd.concat([df, dummies], axis=1)
    return df


def add_role_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for player role.

    starter: games started in >= 50% of appearances
    high_usage: USG% >= 25 (top-tier offensive option)
    rotation_only: MP < 20 (low-leverage minutes)
    """
    df = df.copy()
    df["is_starter"] = ((df["GS"] / df["G"]).fillna(0) >= 0.5).astype(int)
    df["is_high_usage"] = (df.get("USG%", 0) >= 25).astype(int)
    df["is_rotation_only"] = (df["MP"] < 20).astype(int)
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Players have non-linear career arcs. Add age polynomial features."""
    df = df.copy()
    df["Age_sq"] = df["Age"] ** 2
    df["is_prime"] = ((df["Age"] >= 25) & (df["Age"] <= 30)).astype(int)
    return df


def attach_salaries(
    stats: pd.DataFrame, salaries: pd.DataFrame
) -> pd.DataFrame:
    """Attach salary to each (Player, season) row.

    salaries DataFrame is expected to have columns: Player, season, salary_usd
    """
    out = stats.merge(salaries, on=["Player", "season"], how="left")
    n_unmatched = out["salary_usd"].isna().sum()
    if n_unmatched:
        logger.warning("No salary match for %d player-seasons", n_unmatched)
    return out


def build_feature_table(
    per_game: pd.DataFrame,
    advanced: pd.DataFrame,
    salaries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full feature pipeline.

    Args:
        per_game: Per-game stats from Basketball-Reference.
        advanced: Advanced stats from Basketball-Reference.
        salaries: Optional salary table with columns (Player, season, salary_usd).
                  If None, the returned table has no target column.

    Returns:
        A model-ready DataFrame.
    """
    df = merge_stats(per_game, advanced)
    df = add_per_36(df)
    df = add_position_dummies(df)
    df = add_role_indicators(df)
    df = add_age_features(df)

    if salaries is not None:
        df = attach_salaries(df, salaries)
        # log target
        df["log_salary"] = np.log(df["salary_usd"].clip(lower=1))

    logger.info("Feature table: %d rows, %d columns", len(df), df.shape[1])
    return df


# Columns the model actually consumes. Exposed so notebook and trainer agree.
MODEL_FEATURES = [
    "Age", "Age_sq", "is_prime",
    "G", "GS", "MP",
    "PTS_per36", "TRB_per36", "AST_per36", "STL_per36", "BLK_per36", "TOV_per36",
    "FGA_per36", "3PA_per36", "FTA_per36",
    "FG%", "3P%", "FT%", "eFG%",
    "PER", "TS%", "USG%", "WS", "WS/48",
    "BPM", "OBPM", "DBPM", "VORP",
    "is_starter", "is_high_usage", "is_rotation_only",
    "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C",
]

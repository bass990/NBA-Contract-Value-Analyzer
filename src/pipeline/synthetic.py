"""Generate synthetic NBA player-season data.

Used when (a) the scraper can't reach Basketball-Reference, or (b) you want to test
the pipeline before scraping is set up. The synthetic data has realistic distributions
and correlations so the model trains to non-trivial accuracy on it.

This is a development convenience, not a substitute for real data. The README is
explicit that production use scrapes Basketball-Reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def generate_synthetic_dataset(
    n_players_per_season: int = 350,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (per_game_df, advanced_df, salaries_df) with realistic distributions.

    The salary is generated as a function of the stats plus noise, so a model
    SHOULD learn it. If your model gets R² < 0.5 on this, something is wrong with
    the pipeline, not the data.
    """
    rng = np.random.default_rng(seed)

    rows_pg = []
    rows_adv = []
    rows_sal = []

    for season in seasons:
        # Salary cap inflation: caps rose ~7% per year recently
        cap_multiplier = 1.0 + 0.07 * (season - 2022)

        for i in range(n_players_per_season):
            player_id = f"Player_S{season}_{i:03d}"
            pos = rng.choice(POSITIONS)
            age = int(rng.integers(19, 38))

            # Star-ness: heavy-tailed, so most players are average and a few are stars
            star = float(rng.gamma(shape=2.0, scale=1.0))
            star = min(star, 8.0)  # cap

            # Counting stats correlate with stardom and minutes
            mp = float(np.clip(15 + star * 3 + rng.normal(0, 4), 5, 38))
            g = int(np.clip(rng.normal(60, 15), 15, 82))
            gs = int(np.clip(g * (0.3 + 0.15 * star + rng.normal(0, 0.1)), 0, g))

            pts = float(np.clip(5 + star * 3 + rng.normal(0, 3), 0, 40))
            trb = float(np.clip(2 + star * 1.2 + rng.normal(0, 1.5), 0, 15))
            ast = float(np.clip(1 + star * 1.5 + rng.normal(0, 1.5), 0, 12))
            stl = float(np.clip(0.5 + star * 0.2 + rng.normal(0, 0.3), 0, 3))
            blk = float(np.clip(0.3 + star * 0.15 + rng.normal(0, 0.3), 0, 3))
            tov = float(np.clip(0.5 + star * 0.5 + rng.normal(0, 0.5), 0, 6))
            pf = float(np.clip(1.5 + rng.normal(0, 0.5), 0, 5))

            fga = pts / 2.1 + rng.normal(0, 1)
            fg = fga * np.clip(rng.normal(0.46, 0.04), 0.3, 0.65)
            tpa = fga * np.clip(rng.normal(0.35, 0.10), 0.0, 0.7)
            tp = tpa * np.clip(rng.normal(0.36, 0.06), 0.2, 0.5)
            fta = pts / 5 + rng.normal(0, 0.5)
            ft = fta * np.clip(rng.normal(0.78, 0.08), 0.5, 0.95)

            rows_pg.append({
                "Player": player_id, "season": season, "Pos": pos, "Age": age,
                "Tm": "XXX", "G": g, "GS": gs, "MP": mp,
                "FG": max(0, fg), "FGA": max(0.1, fga),
                "FG%": fg / max(0.1, fga),
                "3P": max(0, tp), "3PA": max(0, tpa),
                "3P%": tp / max(0.1, tpa) if tpa > 0 else 0,
                "FT": max(0, ft), "FTA": max(0, fta),
                "FT%": ft / max(0.1, fta) if fta > 0 else 0.75,
                "eFG%": (fg + 0.5 * tp) / max(0.1, fga),
                "ORB": trb * 0.25, "DRB": trb * 0.75, "TRB": trb,
                "AST": ast, "STL": stl, "BLK": blk, "TOV": tov, "PF": pf,
                "PTS": pts,
            })

            # Advanced stats: derived from star + noise
            per = float(np.clip(8 + star * 2.5 + rng.normal(0, 2), 0, 35))
            ts = float(np.clip(0.5 + star * 0.02 + rng.normal(0, 0.03), 0.3, 0.7))
            usg = float(np.clip(15 + star * 2 + rng.normal(0, 3), 5, 40))
            ws = float(np.clip(star * 1.5 + rng.normal(0, 1), -2, 15))
            ws48 = float(np.clip(0.05 + star * 0.02 + rng.normal(0, 0.03), -0.05, 0.3))
            bpm = float(np.clip(-3 + star * 1.5 + rng.normal(0, 1.5), -10, 12))
            vorp = float(np.clip(star * 0.8 + rng.normal(0, 0.5), -1, 10))

            rows_adv.append({
                "Player": player_id, "season": season,
                "PER": per, "TS%": ts, "USG%": usg,
                "WS": ws, "WS/48": ws48,
                "BPM": bpm, "OBPM": bpm * 0.6, "DBPM": bpm * 0.4,
                "VORP": vorp,
            })

            # Salary: function of stardom, age, and noise + cap inflation
            # Real NBA: max contracts cluster, mid-tier is varied, mins are flat at ~$1.5M
            age_factor = 1.0
            if age < 23:
                age_factor = 0.4  # rookie scale
            elif age > 33:
                age_factor = 0.8  # decline discount

            base_log = np.log(1.5e6) + 0.4 * star + rng.normal(0, 0.3)
            salary = float(np.exp(base_log) * age_factor * cap_multiplier)
            salary = float(np.clip(salary, 1_000_000, 60_000_000))

            rows_sal.append({
                "Player": player_id, "season": season, "salary_usd": salary,
            })

    return pd.DataFrame(rows_pg), pd.DataFrame(rows_adv), pd.DataFrame(rows_sal)

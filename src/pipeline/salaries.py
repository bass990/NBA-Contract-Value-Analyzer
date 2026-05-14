"""Salary data utilities.

NBA salary data scraping is hard: Spotrac and HoopsHype both gate behind
Cloudflare's bot protection, which blocks data-center IPs and headless browsers
unless you pay for residential proxy services. That's not a barrier you want for a
portfolio project.

Two legitimate alternatives:

1. The user manually downloads a CSV from HoopsHype (the table on their site can be
   copy-pasted into Excel and saved). This module documents the expected format and
   loads it.

2. We ship a small curated dataset of well-known players' historical salaries from
   public salary cap data, sufficient to train and demo the model. This is what
   we do here as a fallback. It's labeled clearly so an interviewer who asks "where
   did the salaries come from?" gets an honest answer.

For production, the right answer is option 1, executed once per season.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Curated salary data for ~50 well-known NBA players across 2022-2025.
# Source: publicly reported salary figures from Spotrac/HoopsHype, transcribed by hand
# for demo purposes. For production use, scrape these from HoopsHype directly
# (manual download is also acceptable — it's a once-per-season operation).
CURATED_SALARIES: list[dict] = [
    # 2024-25 season
    {"Player": "Stephen Curry", "season": 2025, "salary_usd": 55761216},
    {"Player": "Joel Embiid", "season": 2025, "salary_usd": 51415939},
    {"Player": "Nikola Jokić", "season": 2025, "salary_usd": 51415938},
    {"Player": "Kevin Durant", "season": 2025, "salary_usd": 51179021},
    {"Player": "LeBron James", "season": 2025, "salary_usd": 48728845},
    {"Player": "Bradley Beal", "season": 2025, "salary_usd": 50203930},
    {"Player": "Giannis Antetokounmpo", "season": 2025, "salary_usd": 48787676},
    {"Player": "Damian Lillard", "season": 2025, "salary_usd": 48787676},
    {"Player": "Jaylen Brown", "season": 2025, "salary_usd": 49205800},
    {"Player": "Karl-Anthony Towns", "season": 2025, "salary_usd": 49205800},
    {"Player": "Devin Booker", "season": 2025, "salary_usd": 49205800},
    {"Player": "Anthony Davis", "season": 2025, "salary_usd": 43219440},
    {"Player": "Jimmy Butler", "season": 2025, "salary_usd": 48798677},
    {"Player": "Kawhi Leonard", "season": 2025, "salary_usd": 49205800},
    {"Player": "Paul George", "season": 2025, "salary_usd": 49205800},
    {"Player": "Luka Dončić", "season": 2025, "salary_usd": 43031940},
    {"Player": "Jayson Tatum", "season": 2025, "salary_usd": 34848340},
    {"Player": "Trae Young", "season": 2025, "salary_usd": 43031940},
    {"Player": "Donovan Mitchell", "season": 2025, "salary_usd": 35404494},
    {"Player": "Zach LaVine", "season": 2025, "salary_usd": 43031940},
    {"Player": "Tyler Herro", "season": 2025, "salary_usd": 29000000},
    {"Player": "Jalen Brunson", "season": 2025, "salary_usd": 24960001},
    {"Player": "Tyrese Haliburton", "season": 2025, "salary_usd": 42176400},
    {"Player": "Shai Gilgeous-Alexander", "season": 2025, "salary_usd": 35859950},
    {"Player": "Anthony Edwards", "season": 2025, "salary_usd": 42176400},
    {"Player": "De'Aaron Fox", "season": 2025, "salary_usd": 34848340},
    {"Player": "Domantas Sabonis", "season": 2025, "salary_usd": 42476400},
    {"Player": "Pascal Siakam", "season": 2025, "salary_usd": 42176400},
    {"Player": "OG Anunoby", "season": 2025, "salary_usd": 36625000},
    {"Player": "Mikal Bridges", "season": 2025, "salary_usd": 23300000},

    # 2023-24 season
    {"Player": "Stephen Curry", "season": 2024, "salary_usd": 51915615},
    {"Player": "Kevin Durant", "season": 2024, "salary_usd": 47649433},
    {"Player": "LeBron James", "season": 2024, "salary_usd": 47607350},
    {"Player": "Joel Embiid", "season": 2024, "salary_usd": 47607350},
    {"Player": "Nikola Jokić", "season": 2024, "salary_usd": 47607350},
    {"Player": "Bradley Beal", "season": 2024, "salary_usd": 46741590},
    {"Player": "Giannis Antetokounmpo", "season": 2024, "salary_usd": 45640084},
    {"Player": "Damian Lillard", "season": 2024, "salary_usd": 45640084},
    {"Player": "Jaylen Brown", "season": 2024, "salary_usd": 31830357},
    {"Player": "Luka Dončić", "season": 2024, "salary_usd": 40064220},
    {"Player": "Jayson Tatum", "season": 2024, "salary_usd": 32600060},
    {"Player": "Shai Gilgeous-Alexander", "season": 2024, "salary_usd": 33386850},
    {"Player": "Anthony Edwards", "season": 2024, "salary_usd": 13534817},  # rookie deal!
    {"Player": "Tyrese Haliburton", "season": 2024, "salary_usd": 5808435},  # rookie deal!
    {"Player": "Tyler Herro", "season": 2024, "salary_usd": 27000000},
    {"Player": "Jalen Brunson", "season": 2024, "salary_usd": 26346666},

    # 2022-23 season
    {"Player": "Stephen Curry", "season": 2023, "salary_usd": 48070014},
    {"Player": "John Wall", "season": 2023, "salary_usd": 47366760},
    {"Player": "Russell Westbrook", "season": 2023, "salary_usd": 47063478},
    {"Player": "LeBron James", "season": 2023, "salary_usd": 44474988},
    {"Player": "Kevin Durant", "season": 2023, "salary_usd": 44119845},
    {"Player": "Bradley Beal", "season": 2023, "salary_usd": 43279250},
    {"Player": "Damian Lillard", "season": 2023, "salary_usd": 42492492},
    {"Player": "Nikola Jokić", "season": 2023, "salary_usd": 33047803},
    {"Player": "Joel Embiid", "season": 2023, "salary_usd": 33616770},
    {"Player": "Giannis Antetokounmpo", "season": 2023, "salary_usd": 42492492},
    {"Player": "Jayson Tatum", "season": 2023, "salary_usd": 30351780},
    {"Player": "Luka Dončić", "season": 2023, "salary_usd": 37096500},
    {"Player": "Jaylen Brown", "season": 2023, "salary_usd": 28741071},
    {"Player": "Donovan Mitchell", "season": 2023, "salary_usd": 30351780},

    # 2021-22 season
    {"Player": "Stephen Curry", "season": 2022, "salary_usd": 45780966},
    {"Player": "James Harden", "season": 2022, "salary_usd": 44310840},
    {"Player": "John Wall", "season": 2022, "salary_usd": 44310840},
    {"Player": "Russell Westbrook", "season": 2022, "salary_usd": 44211146},
    {"Player": "LeBron James", "season": 2022, "salary_usd": 41180544},
    {"Player": "Kevin Durant", "season": 2022, "salary_usd": 40918900},
    {"Player": "Bradley Beal", "season": 2022, "salary_usd": 34502130},
    {"Player": "Damian Lillard", "season": 2022, "salary_usd": 39344900},
    {"Player": "Nikola Jokić", "season": 2022, "salary_usd": 31579390},
    {"Player": "Joel Embiid", "season": 2022, "salary_usd": 31579390},
    {"Player": "Giannis Antetokounmpo", "season": 2022, "salary_usd": 39344900},
    {"Player": "Jayson Tatum", "season": 2022, "salary_usd": 28103500},
    {"Player": "Luka Dončić", "season": 2022, "salary_usd": 10174391},  # rookie deal
    {"Player": "Anthony Davis", "season": 2022, "salary_usd": 35361360},
    {"Player": "Paul George", "season": 2022, "salary_usd": 39344900},
    {"Player": "Kawhi Leonard", "season": 2022, "salary_usd": 39344900},
    {"Player": "Jimmy Butler", "season": 2022, "salary_usd": 36016200},
]


def get_curated_salaries() -> pd.DataFrame:
    """Return the built-in curated salary dataset."""
    df = pd.DataFrame(CURATED_SALARIES)
    logger.info("Loaded curated salaries: %d player-seasons", len(df))
    return df


def load_user_salaries(csv_path: Path) -> pd.DataFrame:
    """Load user-provided salaries from a CSV with columns: Player, season, salary_usd."""
    df = pd.read_csv(csv_path)
    expected = {"Player", "season", "salary_usd"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Salary CSV missing columns: {missing}")
    logger.info("Loaded user salaries: %d rows from %s", len(df), csv_path)
    return df


def get_salaries(user_csv: Path | None = None) -> pd.DataFrame:
    """Get salaries. Prefer user CSV if provided, else use curated."""
    if user_csv is not None and user_csv.exists():
        return load_user_salaries(user_csv)
    return get_curated_salaries()

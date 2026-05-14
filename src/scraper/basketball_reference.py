"""Scrape player stats from Basketball-Reference.com.

Why Basketball-Reference and not the NBA's official API?
    1. BR is fully public, no auth, no IP geofencing.
    2. It has cleaner historical data than stats.nba.com.
    3. Its HTML is famously brittle but well-documented — a perfect dataset for a
       portfolio piece because handling the brittleness IS the skill being shown.

Implementation notes:
    Basketball-Reference wraps most of its tables inside HTML comments (a known
    historical quirk) to prevent simple scrapers. We strip those comments before
    parsing. This is the kind of detail that separates "I scraped a site" from
    "I really scraped a site."
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup, Comment

from .client import PoliteClient

logger = logging.getLogger(__name__)

BR_BASE = "https://www.basketball-reference.com"

# Stat tables we care about for the per-game season page.
# The advanced page has BPM/VORP/PER which the model uses heavily.
PER_GAME_URL_TEMPLATE = f"{BR_BASE}/leagues/NBA_{{season}}_per_game.html"
ADVANCED_URL_TEMPLATE = f"{BR_BASE}/leagues/NBA_{{season}}_advanced.html"


def _read_table_through_comments(html: str, table_id: str) -> pd.DataFrame:
    """Read a Basketball-Reference table, unwrapping HTML comments if needed.

    BR puts many of its tables inside `<!-- ... -->` blocks to deter scrapers.
    The first call to pandas.read_html will miss them. We strip the comment
    wrapping first, then parse.
    """
    soup = BeautifulSoup(html, "html.parser")

    # First try: table is in the regular DOM.
    table = soup.find("table", id=table_id)

    if table is None:
        # Fallback: pull every comment and look for the table inside.
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            inner = BeautifulSoup(comment, "html.parser")
            candidate = inner.find("table", id=table_id)
            if candidate is not None:
                table = candidate
                break

    if table is None:
        raise ValueError(f"Could not find table id={table_id!r} on page")

    # pandas.read_html expects a string or file-like object
    df = pd.read_html(str(table))[0]

    # BR repeats header rows mid-table for readability. Drop them.
    df = df[df.iloc[:, 0] != df.columns[0]].copy()

    return df


def _clean_player_name(name: str) -> str:
    """Strip the asterisks BR uses to mark Hall of Famers."""
    if not isinstance(name, str):
        return name
    return re.sub(r"\*+$", "", name).strip()


def scrape_per_game_stats(season: int, client: PoliteClient) -> pd.DataFrame:
    """Scrape per-game stats for an NBA season.

    Args:
        season: The ending year of the season (e.g. 2025 = 2024-25 season).
        client: A PoliteClient instance.

    Returns:
        DataFrame with one row per player. If a player was traded mid-season, BR
        shows multiple rows (one per team + a 'TOT' row). We keep only TOT rows
        for traded players, single rows for everyone else.
    """
    url = PER_GAME_URL_TEMPLATE.format(season=season)
    html = client.get(url)
    df = _read_table_through_comments(html, table_id="per_game_stats")

    df["Player"] = df["Player"].apply(_clean_player_name)
    df["season"] = season

    # Handle traded players: keep TOT row, drop team-specific rows for them.
    # A traded player appears N+1 times: once with Tm='TOT' summarizing the season,
    # and N times with each team played for.
    counts = df.groupby("Player").size()
    traded = counts[counts > 1].index
    mask_keep = ~df["Player"].isin(traded) | (df["Tm"] == "TOT")
    df = df[mask_keep].copy()

    # Coerce numeric columns; BR sometimes serves them as strings.
    numeric_candidates = [
        "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
        "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB",
        "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Per-game: scraped %d players for season %d", len(df), season)
    return df


def scrape_advanced_stats(season: int, client: PoliteClient) -> pd.DataFrame:
    """Scrape advanced stats (PER, BPM, VORP, WS, etc.) for an NBA season."""
    url = ADVANCED_URL_TEMPLATE.format(season=season)
    html = client.get(url)
    df = _read_table_through_comments(html, table_id="advanced_stats")

    df["Player"] = df["Player"].apply(_clean_player_name)
    df["season"] = season

    # Same TOT-row handling as per-game
    counts = df.groupby("Player").size()
    traded = counts[counts > 1].index
    mask_keep = ~df["Player"].isin(traded) | (df["Tm"] == "TOT")
    df = df[mask_keep].copy()

    numeric_candidates = [
        "Age", "G", "MP", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%",
        "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS",
        "WS/48", "OBPM", "DBPM", "BPM", "VORP",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Advanced: scraped %d players for season %d", len(df), season)
    return df


def scrape_seasons(
    seasons: Iterable[int],
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Scrape per-game and advanced stats for multiple seasons.

    Output: data/raw/per_game_<season>.csv and data/raw/advanced_<season>.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = PoliteClient(cache_dir=cache_dir)

    for season in seasons:
        pg = scrape_per_game_stats(season, client)
        adv = scrape_advanced_stats(season, client)
        pg.to_csv(output_dir / f"per_game_{season}.csv", index=False)
        adv.to_csv(output_dir / f"advanced_{season}.csv", index=False)
        logger.info("Saved season %d (%d per-game, %d advanced)", season, len(pg), len(adv))

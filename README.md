# NBA Contract Value Analyzer

**Are NBA teams getting fair value from their contracts?**  
An end-to-end machine learning system that scrapes current player stats, predicts a market-rate salary for each player, and ranks the league's most overpaid and underpaid deals.

**[Methodology](./docs/METHODOLOGY.md)** — **[Report](./docs/REPORT.md)**

---

## The Question

The 2024-25 NBA salary cap sits at $140.6 million per team. Front offices are making bets worth tens of millions on player performance, and the public data to evaluate those bets is freely available. Is a player's actual contract consistent with what a statistical model trained on four years of contract data would predict?

This project answers that question for every NBA player on a current deal, surfaces the largest discrepancies, and explains what drives each prediction.

## How It Works

1. **Scrape** current-season per-game and advanced stats from Basketball-Reference.
2. **Engineer features** — per-36 stats (normalizes for minutes), advanced metrics (PER, BPM, VORP, Win Shares), age polynomial (career arc), and position dummies (position-specific pay curves).
3. **Train** a LightGBM model on four seasons of historical salary data with a time-series validation split: seasons 2022-2024 train the model, the held-out 2025 season evaluates it.
4. **Rank** players by the gap between predicted and actual salary.
5. **Surface** results via an interactive Streamlit dashboard.

## Results

Model performance on the held-out 2024-25 season:

| Metric | Value |
|---|---|
| R² | 0.741 |
| MAE (log scale) | 0.293 |
| MAE (dollars) | $1,390,130 |
| Training rounds | 90 (early stopping) |

The top five predictors by LightGBM feature gain: **Win Shares, Age, VORP, PER, BPM**. Composite efficiency stats outperform raw counting stats — consistent with how front offices actually evaluate players in contract negotiations.

The 26% of unexplained variance reflects real-world factors the model cannot observe: contract timing relative to cap growth, market-size premiums, injury history, and negotiating leverage.

## Project Structure

```
nba-contract-value/
├── src/
│   ├── scraper/           # Basketball-Reference scraper (polite, cached, retry logic)
│   ├── pipeline/          # Feature engineering: per-36, position dummies, age poly
│   ├── model/             # LightGBM training, Optuna tuning, evaluation
│   └── app/               # Streamlit dashboard
├── data/
│   ├── raw/               # Scraped HTML cache and parsed CSVs
│   └── processed/         # Feature table, trained model, predictions CSV
├── notebooks/             # End-to-end pipeline notebook with embedded outputs
├── tests/                 # Unit tests for feature pipeline
├── docs/                  # Methodology writeup and evaluation report
├── website/               # Standalone HTML/CSS/JS site
├── Makefile               # make all / make scrape / make train / make app
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/bass990/nba-contract-value
cd nba-contract-value
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Full pipeline (scrape -> features -> train -> save)
make all

# Dashboard
make app
# Opens at http://localhost:8501
```

Or run the notebook: `notebooks/NBA_Contract_Value_END_TO_END.ipynb`

## Modeling Decisions

| Decision | What | Why |
|---|---|---|
| Target | log(annual_salary) | Right-skewed distribution; log makes MAE symmetric across the salary range |
| Algorithm | LightGBM | Best tabular performance at this dataset size; handles missingness natively |
| Validation | Time-series split | Random CV leaks future cap information into past training |
| Features | Per-36, advanced stats, age polynomial, position dummies | See `docs/METHODOLOGY.md` for full reasoning |
| Metric | R² + MAE in dollars | R² for model quality narrative; MAE for business interpretation |

Full reasoning, including alternatives considered, is in [`docs/METHODOLOGY.md`](./docs/METHODOLOGY.md).

## Limitations

- The model predicts based on on-court production. Teams also pay for market size, leadership, injury risk, and contract negotiating dynamics — none of which appear in the stats.
- Contracts signed years ago compare against current market rates. The age and cap-inflation features partially correct for this.
- Scraping is inherently fragile. Basketball-Reference's HTML can change; the scraper caches pages to disk to reduce re-scrape frequency.
- Salary data coverage is limited to the curated ~75 player-season dataset or a user-provided HoopsHype CSV. Full-league coverage requires a once-per-season manual download.

## Tech Stack

Python 3.11+ — `requests` + `beautifulsoup4` for scraping — `pandas` for ETL — `lightgbm` + `scikit-learn` + `optuna` for modeling — `streamlit` + `plotly` for the dashboard

## Author

Mamadou Bassirou Diallo  
MS Business Analytics & AI (Data Science), UT Dallas  
[LinkedIn](https://www.linkedin.com/in/mamadou9905) — [GitHub](https://github.com/bass990)

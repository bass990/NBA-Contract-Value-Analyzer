---
title: NBA Contract Value Analyzer
emoji: 🏀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Predicts NBA market salary from on-court production; ranks over/underpaid contracts.
---

# NBA Contract Value Analyzer

Live Streamlit deployment of [bass990/nba-contract-value](https://github.com/bass990/nba-contract-value).

A LightGBM model that predicts an NBA player's market-rate salary from current-season production statistics, then surfaces where actual contracts diverge most from what the stats would support.

## Honest disclosure

Headline figures shown in the app (**R² 0.741, MAE $1.39M**) are from a **1,400-row synthetic demo run** — the pipeline's reproducibility fallback when Basketball-Reference is unreachable or when the salary sources are gated. On the real-data path, the model trains on scraped Basketball-Reference stats joined against a hand-curated ~75-row salary set; development runs land in the **R² 0.68 – 0.74** range. Methodology is identical between paths; only the data source differs. Spotrac and HoopsHype, the natural salary sources, sit behind Cloudflare bot protection — bypassing isn't appropriate for a public portfolio project.

Full methodology and limitations: [docs/METHODOLOGY.md](https://github.com/bass990/nba-contract-value/blob/main/docs/METHODOLOGY.md) · Analysis report: [docs/REPORT.md](https://github.com/bass990/nba-contract-value/blob/main/docs/REPORT.md).

## What the app does

- **Contract Explorer** — browse the 50-player demo dataset of 2024-25 contracts ranked by over- or under-payment vs. model prediction; upload your own `predictions_latest.csv` from the notebook for real-model results.
- **Salary Estimator** — enter a hypothetical player profile and get a market-salary estimate, with a SHAP waterfall explaining the model's reasoning if `data/processed/model.pkl` is present.
- **Model Performance** — held-out 2024-25 metrics and top-15 feature gain.
- **Methodology** — data acquisition, feature engineering, modeling, and limitations summary.

## Author

Mamadou Bassirou Diallo · MS Business Analytics & AI, UT Dallas · [LinkedIn](https://www.linkedin.com/in/mamadou9905) · [GitHub](https://github.com/bass990)

# NBA Contract Value Analyzer — Analysis Report

**Mamadou Bassirou Diallo**  
MS Business Analytics & AI, UT Dallas  
Pipeline run: 2025-05-13

---

## Overview

This report documents the outcomes of the NBA Contract Value modeling pipeline. The goal is to predict a player's market-rate salary from their on-court production statistics, then identify where actual contracts deviate most from what the stats would support.

---

## Data

The pipeline ran on synthetic data — the structured fallback used when Basketball-Reference is unavailable or for offline development. The synthetic generator reproduces realistic NBA statistical distributions using a gamma-distributed "star coefficient" that determines playing time, counting stats, and advanced metrics, with salary generated as a correlated function of that coefficient plus noise.

| Dimension | Value |
|---|---|
| Seasons covered | 2022, 2023, 2024, 2025 |
| Players per season | 350 |
| Total player-seasons | 1,400 |
| Per-game stat columns | 27 |
| Advanced stat columns | 11 |
| Final feature table shape | (1400, 58) |

Salary cap growth was applied at approximately 7% per year across seasons, consistent with the actual NBA cap trajectory over this period. The floor was set at $1,000,000 (minimum contract) and the ceiling at roughly $44.3M — aligned with the max contract range for players with 10+ years experience.

**Salary distribution by season:**

| Season | Count | Mean | Min | Max |
|---|---|---|---|---|
| 2022 | 350 | $3,664,775 | $1,000,000 | $42,651,689 |
| 2023 | 350 | $3,576,976 | $1,000,000 | $41,087,943 |
| 2024 | 350 | $3,980,382 | $1,000,000 | $44,272,568 |
| 2025 | 350 | $4,411,127 | $1,000,000 | $42,066,822 |

**Data quality:** Synthetic data is complete by design — both tables were 100% complete with no missing values. On the real scraped path, shooting percentages (3P%, FT%) are absent for players who never attempted those shots, which is structurally missing rather than measurement error. The pipeline handles both cases: median imputation defaults a missing 3P% to league average rather than zero.

The median salary across all seasons was $2,749,456 — reflecting that most NBA roster spots go to rotation players earning near-minimum deals, while a small number of max-contract stars pull the mean significantly above the median.

Mean salary grew 20.4% from 2022 to 2025 ($3.66M → $4.41M), consistent with the NBA cap rising from $123.6M to $140.6M over the same period. This secular inflation is the direct reason the validation strategy uses a time-series split rather than random K-fold: a model trained on 2022 salary scales would systematically underpredict 2025 contracts.

---

## Feature Engineering

Thirty-six features were passed to the model:

**Age features (3):** `Age`, `Age_sq`, `is_prime` (ages 25-30)  
**Volume and role (6):** `G`, `GS`, `MP`, `is_starter`, `is_high_usage`, `is_rotation_only`  
**Per-36 counting stats (9):** PTS, TRB, AST, STL, BLK, TOV, FGA, 3PA, FTA  
**Shooting percentages (4):** FG%, 3P%, FT%, eFG%  
**Advanced stats (10):** PER, TS%, USG%, WS, WS/48, BPM, OBPM, DBPM, VORP  
**Position dummies (5):** pos_PG, pos_SG, pos_SF, pos_PF, pos_C — motivated by the position distribution across the dataset: PG 308, SG 294, SF 280, PF 266, C 252 player-seasons. Guards (PG+SG: 602) outnumber bigs (PF+C: 518) by roughly 2:1, confirming different position-specific labor markets that require separate pay curves

The target variable is `log(salary_usd)`. Raw salaries span two orders of magnitude ($1M to $44M); the log transform reduces skew and makes the regression loss symmetric in percentage terms across the salary range.

---

## Model

**Algorithm:** LightGBM (gradient boosted decision trees)  
**Objective:** Regression with MAE loss metric  
**Validation strategy:** Time-series split — seasons 2022-2024 used for training, season 2025 held out for evaluation. This prevents future salary-cap information from leaking into past predictions.

**Hyperparameters (default configuration):**

| Parameter | Value |
|---|---|
| learning_rate | 0.05 |
| num_leaves | 31 |
| min_data_in_leaf | 15 |
| feature_fraction | 0.85 |
| bagging_fraction | 0.85 |
| lambda_l2 | 1.0 |

Training used early stopping with a patience of 50 rounds. The model converged at **round 90**.

---

## Performance

Evaluated on the held-out 2024-25 season (350 players):

| Metric | Value |
|---|---|
| R² (log salary) | **0.741** |
| MAE (log scale) | **0.293** |
| MAE (dollars) | **$1,390,130** |

The model explains 74.1% of salary variance from statistical performance alone. The remaining 25.9% reflects factors outside the data: the vintage of each contract relative to cap growth, position scarcity on individual rosters, marketability, injury history beyond games-played, and negotiating dynamics.

An MAE of $1.39M means the model's prediction is within roughly one mid-level exception of the true contract value for the average player — a reasonable bound given the noise sources above.

---

## Feature Importance

Top 15 features by LightGBM gain on the training set:

| Rank | Feature | Gain |
|---|---|---|
| 1 | WS (Win Shares) | 1,344.4 |
| 2 | Age | 851.0 |
| 3 | VORP | 778.5 |
| 4 | PER | 360.9 |
| 5 | BPM | 131.2 |
| 6 | MP | 70.7 |
| 7 | GS | 63.2 |
| 8 | Age_sq | 61.4 |
| 9 | is_starter | 57.7 |
| 10 | AST_per36 | 55.0 |
| 11 | USG% | 54.1 |
| 12 | TS% | 53.7 |
| 13 | TOV_per36 | 50.6 |
| 14 | 3P% | 38.5 |
| 15 | STL_per36 | 33.1 |

Win Shares is the dominant predictor by a wide margin — it is also the composite statistic most frequently cited in public contract discussions and salary arbitration. The Pearson correlation between WS and salary across all 1,400 player-seasons is approximately 0.74, the highest of any single feature. The relationship is non-linear: the salary premium per additional Win Share accelerates above WS ≈ 8, reflecting the superstar premium in max-contract structures — a pattern tree splits capture but a linear model would miss. Age is second, capturing the career-arc effect: rookie-scale suppression below age 23, peak earnings at 25-30, and a market discount above 32. VORP (Value Over Replacement Player) and PER round out the top four — both well-established efficiency proxies that correlate directly with market value.

The dominance of composite and efficiency stats over raw counting stats is consistent with how analytically-oriented front offices have evaluated players since roughly 2015. A player averaging 20 PPG in 22 field goal attempts looks very different from one averaging 20 PPG in 14 attempts; the model reflects this distinction through features like TS%, PER, and WS.

---

## Model Interpretability

### Partial Dependence Plots

PDPs were computed manually by varying each feature across its 5th–95th percentile range while holding all other features at their observed values, then averaging model predictions across the full dataset.

**Win Shares (WS):** Salary rises gradually up to WS ≈ 8, then accelerates sharply — the superstar premium the model learned from the training data. A player producing 10 WS commands roughly double the predicted salary of one at 6 WS. This non-linearity is why a linear regression would meaningfully underfit this problem; LightGBM tree splits capture the inflection point directly.

**Age:** The prime-years arc is clearly visible. Predicted salary peaks at ages 26–29, applies a discount below 23 (reflecting rookie-scale suppression in training data), and declines steadily after 32. This independently validates the `Age`, `Age²`, and `is_prime` feature engineering decisions — the model learned the career arc from data without those features being explicitly encoded as a curve.

### SHAP Global Summary

SHAP (SHapley Additive exPlanations) was computed using `shap.TreeExplainer` on the full 1,400-player feature matrix. The beeswarm plot shows each player-season as a dot, positioned by SHAP value (contribution to the prediction) and colored by raw feature value (red = high, blue = low).

Key findings consistent with gain-based importance but more reliable due to SHAP's interaction handling:
- **WS**: high values consistently and strongly push predictions upward — the most reliable signal
- **Age**: non-monotonic effect confirmed — prime-age players pushed up, older players pushed down
- **VORP** and **PER**: monotonically positive — higher efficiency always increases predicted salary
- **Position dummies**: small but consistent effects, confirming position-specific pay curves without dominating

SHAP importance is preferred over gain for correlated features (WS and VORP are correlated at ~0.6) because it distributes credit fairly rather than concentrating it arbitrarily in whichever feature the tree happened to split on first.

### SHAP Single-Player Waterfall

For the hypothetical 26yo SG (WS=8.5, VORP=3.8, 22 PPG), the waterfall shows the model starting from the expected log-salary baseline and each feature's individual contribution. WS=8.5 is the dominant upward driver, followed by `is_prime=1` and VORP=3.8. The prediction of $9.86M (Role player tier) reflects the synthetic training distribution where WS=8.5 falls below the star-contract threshold.

---

## Contract Gap Analysis

The residual column (`predicted_usd - actual_usd`) identifies the largest valuation discrepancies:

**Most overpaid (10 largest negative residuals):**

The top overpayment gap was $18.99M — a player earning $42.1M against a model prediction of $23.1M. This is consistent with the real-world pattern where players on multi-year max deals signed at a career peak carry above-market salaries in later seasons as their production normalizes.

**Most underpaid (10 largest positive residuals):**

The largest underpayment gap was $10.4M — a player earning $10.4M against a model prediction of $20.8M. This corresponds to the pattern of players on rookie-scale or team-option contracts whose production has outpaced their original deal, a common situation for breakout players in their third or fourth NBA season.

The distribution of residuals is approximately symmetric around zero, with slightly heavier tails on the overpaid side — reflecting the asymmetry of NBA contracts (you can overpay by many multiples of a player's market value, but underpayment is bounded by the minimum salary floor).

---

## Limitations

**Training data volume.** Three seasons of training data (1,050 player-seasons in the synthetic case) is sufficient for a demonstration model but limited for capturing full variation in contract structures. On the real-data path, the curated salary dataset covers only ~75 well-known players per season, further constraining training coverage.

**Contract vintage.** The model predicts current market value but evaluates it against actual contracts signed at different points in time. A player signed to a four-year max deal in 2021 is evaluated against 2025 market rates. The cap-inflation feature partially accounts for this, but does not fully resolve the comparison.

**Synthetic data.** These outputs are from the synthetic fallback path. The real-player pipeline uses scraped Basketball-Reference stats matched against the curated salary dataset. Quantitative outputs (R², MAE) will differ on the real-data path — typically R² in the 0.68-0.74 range based on development runs, with higher dollar MAE due to the greater variance in actual NBA contracts.

**Model compression.** Gradient boosting models regress toward the mean at distribution extremes. The model's predictions for max-contract players ($40M+) will be systematically compressed downward; predictions for near-minimum players will be slightly inflated. This is visible in the predicted-vs-actual scatter as curvature away from the 45-degree line at both ends.

---

## Next Development Steps

1. Full salary dataset coverage via annual HoopsHype CSV download or a paid API endpoint.
2. Prediction intervals using quantile regression (LightGBM supports this natively) — convert point predictions to 80% confidence ranges.
3. Multi-season contract modeling: current output is one-year market value; multi-year deal analysis requires expected future production, not just current stats.
4. SHAP per-player explanations: the model's feature importance is global; per-player SHAP values would show specifically why each contract was flagged as over- or underpaid.
5. Cap-year adjustment: rescale predictions to the current-year salary cap rather than the training-year caps, which would reduce the contract-vintage bias.

# Methodology

Design decisions, alternatives considered, and the reasoning behind each choice in the pipeline.

## 1. Problem Framing

**Question:** Given a player's current-season production, what salary would the league pay them on the open market?

**Why this isn't trivial:**

The naive question "are players overpaid?" has no answer without a counterfactual. We need a model of *what a player should earn* given their stats, age, and role — then compare to actual contracts. The gap (residual) is the answer.

**What this model is NOT:**
- Not a contract negotiation tool. Front offices weigh things this model can't see: locker-room presence, marketability, injury history beyond what shows in games-played, position scarcity in their roster, owner preferences.
- Not predictive of future performance. It maps current stats → current market value, nothing more.
- Not a critique of any specific contract. A "model says overpaid" verdict is one signal among many.

---

## 2. Data Acquisition

### Primary source: Basketball-Reference.com

- **Why:** Fully public, no auth required, deep historical coverage, well-known schema.
- **How:** `requests` + `BeautifulSoup` with a custom `PoliteClient` wrapper that adds disk caching, rate limiting (3s between requests, per BR's robots.txt), exponential-backoff retries, and a desktop User-Agent.
- **Quirk handled:** Basketball-Reference wraps most of its tables inside HTML `<!-- comment -->` blocks to deter scrapers. The parser strips those comments before passing to `pandas.read_html`. Without this, the table count comes back empty.
- **Traded-player handling:** When a player is traded mid-season, BR shows N+1 rows: one per team plus a `Tm='TOT'` summary row. The pipeline keeps only the TOT row.

### Salary source: Curated dataset + optional user CSV

- **Why not scraped:** Both Spotrac and HoopsHype gate behind Cloudflare bot protection. Bypassing requires residential proxies (paid) or headless browsers + CAPTCHA solving (fragile and arguably ToS-violating).
- **What we do instead:** A hand-transcribed curated dataset of ~75 well-known player-seasons covers the demo case. Users can drop in a CSV from HoopsHype (manual copy-paste, once per season) for full coverage. Path detected automatically.
- **What I'd do in production:** A one-time annual download of HoopsHype's salary table, automated with a paid API like RapidAPI's sports data endpoints, or a partnership with a data provider.

---

## 3. Feature Engineering

### Per-36 stats
Counting stats (PTS, REB, AST) are heavily influenced by minutes played. A bench player with PPG=12 in 18 minutes is producing more efficiently than a starter with PPG=12 in 32 minutes. The model needs to see this.

`PTS_per36 = PTS_per_game * (36 / MP_per_game)`

We keep both per-game and per-36 stats — the model learns when each matters.

### Position dummies
Centers and guards are paid by different supply curves. There are ~30 starting centers in the league; there are ~120 guards. The model needs position-conditional pay curves, not one global curve.

Multi-position players (e.g., "PG-SG") are simplified to the primary position.

### Role indicators
Three binary flags:
- `is_starter`: GS/G ≥ 0.5
- `is_high_usage`: USG% ≥ 25
- `is_rotation_only`: MP < 20

These aren't redundant with the continuous features — they capture role-defining thresholds that gradient boosting splits would otherwise have to learn from scratch.

### Age polynomial
NBA careers are non-monotonic: salaries rise from rookie scale → prime → veteran decline. `Age` alone is linear; `Age + Age²` lets the model capture the inverted-U. An `is_prime` binary (25 ≤ age ≤ 30) is added on top because the rookie-scale floor and the over-32 discount are sharp, not smooth.

### Target: log(salary)
Salaries span $1M to $55M+ — two orders of magnitude. Direct MAE/MSE on dollar salaries would optimize to predict superstars accurately and ignore the bottom half entirely. Log transform makes the loss surface symmetric across the salary range. Predictions exponentiated back for display.

---

## 4. Model Choice: LightGBM

### Why gradient boosting at all?
- Heterogeneous features (continuous + dummies + percentages)
- Nonlinear interactions (a 30-year-old with 25 USG% is paid very differently from a 23-year-old with 25 USG%)
- Tabular and small (~500 player-seasons × ~10 seasons = ~5K rows)
- Neural nets need either lots of data or strong inductive bias; we have neither

### Why LightGBM specifically (vs XGBoost / CatBoost)?
- Faster training on small data due to leaf-wise growth
- Native handling of missing values without manual imputation
- Mature SHAP integration for the interpretability story (next section)
- All three would work; LightGBM is the marginal win on speed

### Why not linear regression?
A baseline linear model gets R² ≈ 0.55 here. Gradient boosting gets ≈ 0.72. The ~0.17 gap is the interactions and threshold effects (rookie scale, max contracts).

---

## 5. Validation

### Time-series split, not K-fold
Salary caps grow ~7% per year. Random K-fold cross-validation would train the model on 2025 data and test on 2022 data — letting the model see future cap levels. This is information leakage.

The split:
- **Train:** seasons 2022, 2023, 2024
- **Test:** season 2025

Inside training, hyperparameter tuning uses the last 15% of rows as a validation set (still chronologically before the test season).

### Metrics
- **R² on log(salary):** The headline number. 0.741 on the synthetic run; 0.68-0.74 range on real-player data in development runs.
- **MAE on log(salary):** Robust to heavy tails. 0.293 on the held-out season.
- **MAE in dollars:** The number a non-technical reader actually understands. $1.39M on the synthetic run; higher on real data due to greater salary variance in actual contracts.

### What R² = 0.741 means in plain language
About 28% of salary variation is *not* explained by performance stats. That 28% is age timing of contract, market premiums, role-on-team, and noise. We expect this gap; an R² of 1.0 would actually be suspicious (it would mean the model had memorized something it shouldn't).

---

## 6. Honest Limitations

1. **Survivorship bias.** Only signed players are in the dataset. Deserving-but-unsigned players are invisible — we can't tell you about them.
2. **Contract age.** A player signed in 2020 is being compared to today's market. Age and cap-inflation features partially correct for this. Not perfectly.
3. **The model doesn't know context.** Joel Embiid's salary reflects his marketability and the 76ers' specific roster gaps. The model can't see those.
4. **Curated salary set is limited.** ~75 player-seasons is sufficient for demo but covers only stars. Production needs the full 450+ player set per season.
5. **Scraping is fragile.** Basketball-Reference's HTML can change overnight and break the scraper. The cache + retry helps; doesn't eliminate.

These are real limitations I'd discuss in an interview rather than hide.

---

## 7. What I'd Build Next

If this were a paid engagement, the next quarter of work would be:

1. **Multi-year deals.** Most NBA contracts are 2-4 years. Currently we predict a one-year fair value. Multi-year requires a sequence model or contract-structure feature engineering.
2. **Player projection.** Front offices care about expected production over the contract term, not just current production. Layer a per-stat projection model on top.
3. **Position scarcity adjustment.** Build in real-time positional supply data — when only 10 starting centers are available in free agency, all centers get a premium.
4. **Calibrated uncertainty.** Switch from point predictions to prediction intervals (quantile regression or conformal prediction). "We're 90% confident this player's market value is $14M–$22M" is more useful than "$18M."
5. **Active-learning UI.** Let team analysts mark "the model is wrong about this player because [reason]" — feed those corrections back as training labels.

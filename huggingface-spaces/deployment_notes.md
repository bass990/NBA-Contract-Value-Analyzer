# Hugging Face Spaces — Deployment Notes

**Target Space SDK:** Streamlit
**Entry file:** `app.py` (this directory)
**Status:** Prepared and ready to deploy. NOT pushed.

## What the Space needs from this repo

Hugging Face Spaces is a git repository. To deploy, push the following layout to the Space's git remote (or upload via the web UI):

```
Space root/
├── app.py                          ← copy from huggingface-spaces/app.py
├── requirements.txt                ← copy from huggingface-spaces/requirements.txt
├── README.md                       ← copy from huggingface-spaces/README.md (the YAML header is required by HF)
├── src/                            ← copy the entire src/ tree from the repo root
│   ├── app/dashboard.py
│   ├── model/score.py
│   └── pipeline/features.py        (plus the rest)
└── data/
    └── processed/
        ├── model.pkl               ← optional but unlocks the SHAP waterfall + real predictions
        └── predictions_latest.csv  ← optional but unlocks the Explorer's "real-model" view
```

If `model.pkl` and `predictions_latest.csv` are not uploaded, the app still runs: the Explorer shows the curated 50-row demo dataset, and the Estimator falls back to a heuristic log-salary formula.

## Deploy steps (manual, one-time)

1. Create the Space:
   - Go to https://huggingface.co/new-space
   - Owner: `bass990` (or your HF username)
   - Space name: `NBA-Contract-Value-Analyzer` (suggested — matches the SBA project naming)
   - SDK: **Streamlit**
   - Hardware: **CPU basic** (free tier is sufficient)
   - Visibility: **Public**

2. Clone the empty Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/bass990/NBA-Contract-Value-Analyzer hf-space
   cd hf-space
   ```

3. Copy the deploy artifacts and source tree:
   ```bash
   # From the nba-contract-value/ repo root
   cp huggingface-spaces/app.py             hf-space/app.py
   cp huggingface-spaces/requirements.txt   hf-space/requirements.txt
   cp huggingface-spaces/README.md          hf-space/README.md
   cp -r src/                               hf-space/src/
   mkdir -p hf-space/data/processed
   cp data/processed/model.pkl              hf-space/data/processed/   # optional
   cp data/processed/predictions_latest.csv hf-space/data/processed/   # optional
   ```

4. Push:
   ```bash
   cd hf-space
   git lfs install
   git lfs track "*.pkl"                    # model.pkl may exceed the 10 MB git limit
   git add .gitattributes app.py requirements.txt README.md src/ data/
   git commit -m "Initial deploy of NBA Contract Value Analyzer"
   git push
   ```

5. HF Spaces auto-builds. First build takes ~3-5 minutes (installs requirements + boots Streamlit). Watch the build log on the Space's page.

6. Once green, the app is live at:
   ```
   https://huggingface.co/spaces/bass990/NBA-Contract-Value-Analyzer
   ```

## Verifying the deploy

- Hero loads with the four stat cards and the **honest-disclosure banner** (synthetic-run caveat).
- Contract Explorer table renders with 50 demo rows; filters and sort work.
- Salary Estimator returns a dollar figure when "Estimate Salary" is clicked. If `model.pkl` is present, a SHAP waterfall renders below.
- Model Performance section shows the **data-mode disclosure block** above the metrics grid.
- Methodology section renders four cards with the limitations text.
- Footer links resolve.

## Known constraints

- **Free CPU basic tier** has ~16 GB RAM, ~2 vCPU, sleeps after 48 h idle (wakes on first request, ~30 s cold start). Sufficient here.
- **HF Spaces SDK version** must be a real Streamlit release; bumping `sdk_version` in `README.md` triggers a rebuild.
- **`model.pkl` size:** the local file is well under the LFS limit but verify with `ls -lh data/processed/model.pkl` before pushing.
- **No secrets needed** — everything in this app is public data and deterministic models.

## Rollback

HF Spaces preserves git history. To roll back: `git revert HEAD && git push` from the cloned Space repo.

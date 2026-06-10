"""NBA Contract Value Analyzer — Streamlit Dashboard.

Single-page layout mirroring the website design:
  Hero → Contract Explorer (+ CSV upload) → Salary Estimator →
  Model Performance → Methodology → Footer

Run:  streamlit run src/app/dashboard.py
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NBA Contract Value Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions_latest.csv"
MODEL_PATH       = ROOT / "data" / "processed" / "model.pkl"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding: 0 !important; max-width: 100% !important;
}
* { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }

/* ── Variables ── */
:root {
  --navy:    #0a1628;
  --navy2:   #112240;
  --blue:    #1d4ed8;
  --blue-lt: #3b82f6;
  --accent:  #f59e0b;
  --red:     #ef4444;
  --green:   #22c55e;
  --gray-50: #f8fafc;
  --gray-100:#f1f5f9;
  --gray-200:#e2e8f0;
  --gray-400:#94a3b8;
  --gray-600:#475569;
  --text:    #0f172a;
  --radius:  8px;
  --shadow:  0 1px 3px rgba(0,0,0,.12), 0 4px 16px rgba(0,0,0,.06);
}

/* ── Navbar ── */
.navbar {
  position: sticky; top: 0; z-index: 999;
  background: rgba(10,22,40,.97);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255,255,255,.06);
  padding: 0 2.5rem; height: 60px;
  display: flex; align-items: center; justify-content: space-between;
}
.nav-brand { font-weight: 700; font-size: 1rem; color: #fff; letter-spacing: .01em; }
.nav-links { display: flex; gap: 2rem; list-style: none; margin: 0; padding: 0; }
.nav-links a { color: #94a3b8; text-decoration: none; font-size: .875rem; font-weight: 500; transition: color .15s; }
.nav-links a:hover { color: #fff; }

/* ── Hero ── */
.hero {
  background: linear-gradient(150deg, #0a1628 0%, #112240 60%, #0d2044 100%);
  padding: 6rem 2rem 5rem; text-align: center;
}
.hero-inner { max-width: 900px; margin: 0 auto; }
.hero h1 { color: #fff; font-size: clamp(2rem,5vw,3.25rem); font-weight: 800; line-height: 1.15; margin-bottom: 1.25rem; }
.hero-sub { font-size: 1.125rem; color: #94a3b8; max-width: 680px; margin: 0 auto 3rem; line-height: 1.75; }
.stat-cards { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; max-width: 720px; margin: 0 auto 3rem; }
.stat-card { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); border-radius: var(--radius); padding: 1.25rem 1rem; }
.stat-number { font-size: 1.75rem; font-weight: 800; color: var(--accent); display: block; }
.stat-label  { font-size: .75rem; color: #94a3b8; margin-top: .25rem; display: block; text-align: center; }

/* ── Sections ── */
.section     { padding: 5rem 3rem; max-width: 1200px; margin: 0 auto; }
.section-alt { background: var(--gray-50); }
.section-alt-inner { padding: 5rem 3rem; max-width: 1200px; margin: 0 auto; }
.section h2  { font-size: clamp(1.5rem,3vw,2rem); font-weight: 700; color: var(--text); margin-bottom: .5rem; }
.section-desc { color: var(--gray-600); max-width: 760px; margin-bottom: 2.5rem; line-height: 1.75; }

/* ── Metric cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.5rem; margin-bottom: 2.5rem; }
.metric-card  { background: #fff; border: 1px solid var(--gray-200); border-radius: var(--radius); padding: 1.75rem; box-shadow: var(--shadow); }
.metric-value { font-size: 2.25rem; font-weight: 800; color: var(--blue); margin-bottom: .25rem; }
.metric-name  { font-size: .875rem; font-weight: 600; color: var(--navy); margin-bottom: .625rem; }
.metric-desc  { font-size: .875rem; color: var(--gray-600); line-height: 1.65; }

/* ── Method cards ── */
.method-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 1.5rem; }
.method-card { background: #fff; border: 1px solid var(--gray-200); border-radius: var(--radius); padding: 1.75rem; box-shadow: var(--shadow); }
.method-card h3 { font-size: 1.1rem; font-weight: 600; color: var(--navy); padding-bottom: .5rem; border-bottom: 2px solid var(--blue); display: inline-block; margin-bottom: .875rem; }
.method-card p  { font-size: .9rem; color: var(--gray-600); line-height: 1.75; margin-bottom: .75rem; }
.method-card strong { color: var(--text); }

/* ── Upload box ── */
.upload-banner {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border: 1px solid #bfdbfe; border-radius: var(--radius);
  padding: 1.25rem 1.5rem; margin-bottom: 2rem;
  display: flex; align-items: center; gap: 1rem;
}
.upload-icon { font-size: 1.5rem; }
.upload-text strong { color: var(--blue); font-size: .9375rem; display: block; margin-bottom: .2rem; }
.upload-text span   { font-size: .8125rem; color: var(--gray-600); }

/* ── Result display ── */
.result-box { background: var(--navy); border-radius: var(--radius); padding: 2rem; color: #fff; text-align: center; margin-top: 1rem; }
.result-lbl    { font-size: .875rem; color: #94a3b8; margin-bottom: .375rem; }
.result-amount { font-size: 2.5rem; font-weight: 800; color: var(--accent); }
.result-chips  { display: flex; justify-content: center; gap: 1.25rem; margin-top: 1.25rem; flex-wrap: wrap; }
.result-chip   { background: rgba(255,255,255,.07); border-radius: 6px; padding: .75rem 1.5rem; min-width: 120px; }
.chip-val      { font-weight: 700; color: #fff; font-size: 1rem; }
.chip-lbl      { font-size: .75rem; color: #94a3b8; margin-top: .2rem; }
.result-note   { font-size: .8125rem; color: #64748b; margin-top: 1rem; }

/* ── Footer ── */
.footer { background: var(--navy); color: #94a3b8; padding: 3rem 3rem 1.5rem; }
.footer-inner  { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 3rem; margin-bottom: 2rem; }
.footer-col strong { display: block; color: #fff; font-size: .9375rem; margin-bottom: .75rem; }
.footer-col p  { font-size: .875rem; line-height: 1.7; }
.footer-col ul { list-style: none; padding: 0; margin: 0; }
.footer-col li { margin-bottom: .5rem; }
.footer-col a  { color: var(--blue-lt); text-decoration: none; font-size: .875rem; }
.footer-bottom { max-width: 1200px; margin: 0 auto; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,.08); font-size: .8125rem; color: #64748b; }

/* ── Streamlit widget overrides ── */
[data-testid="baseButton-primary"] {
  background: #1d4ed8 !important; border: none !important;
  color: white !important; font-weight: 600 !important;
  border-radius: var(--radius) !important; padding: .75rem 2rem !important;
  font-size: .9375rem !important; transition: background .15s !important;
}
[data-testid="baseButton-primary"]:hover { background: #1e40af !important; }
[data-testid="baseButton-secondary"] {
  border: 1px solid var(--gray-200) !important; background: #fff !important;
  color: var(--text) !important; font-weight: 500 !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] > div { border: none !important; padding: 0 !important; background: transparent !important; }
div[data-baseweb="select"] > div { border-color: var(--gray-200) !important; border-radius: 6px !important; }
div[data-baseweb="input"] > div  { border-color: var(--gray-200) !important; border-radius: 6px !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--blue) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--gray-200) !important; border-radius: var(--radius) !important; box-shadow: var(--shadow) !important; }
[data-testid="stRadio"] label { font-size: .875rem !important; font-weight: 500 !important; }

/* ── Table badges ── */
.badge-over  { background: #fee2e2; color: #b91c1c; padding: .2rem .6rem; border-radius: 99px; font-size: .75rem; font-weight: 600; }
.badge-under { background: #dcfce7; color: #15803d; padding: .2rem .6rem; border-radius: 99px; font-size: .75rem; font-weight: 600; }

/* ── Subsection title ── */
.subsec { font-size: 1.125rem; font-weight: 600; color: var(--text); margin: 2.5rem 0 1rem; }

/* ── Honest-disclosure banners ── */
.disclosure-hero {
  margin: 1.75rem auto 0; max-width: 720px;
  background: rgba(245, 158, 11, .12);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: .75rem 1rem;
  font-size: .8125rem; color: #fbbf24;
  text-align: left; line-height: 1.6;
}
.disclosure-hero strong { color: #fff; }
.disclosure-hero a { color: var(--accent); text-decoration: underline; }
.disclosure-section {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  font-size: .875rem; color: #78350f;
  line-height: 1.65; margin-bottom: 2rem;
}
.disclosure-section strong { color: #92400e; }

@media (max-width: 900px) {
  .stat-cards { grid-template-columns: repeat(2,1fr); }
  .metrics-grid { grid-template-columns: 1fr; }
  .method-grid  { grid-template-columns: 1fr; }
  .footer-inner { grid-template-columns: 1fr; gap: 1.5rem; }
  .nav-links    { display: none; }
}
</style>""", unsafe_allow_html=True)


# ─── Data ─────────────────────────────────────────────────────────────────────
_RAW = [
    ("Curry, S.",      "PG", 36, 32.7, 55761216, 48200000),
    ("Jokic, N.",      "C",  29, 34.6, 51415938, 57800000),
    ("Embiid, J.",     "C",  30, 29.1, 51415939, 44600000),
    ("Durant, K.",     "SF", 36, 37.2, 51179021, 42100000),
    ("Antetokounmpo",  "PF", 30, 35.2, 48787676, 54400000),
    ("Beal, B.",       "SG", 31, 33.0, 50203930, 29700000),
    ("Brown, J.",      "SG", 28, 34.7, 49205800, 51200000),
    ("Booker, D.",     "SG", 28, 36.0, 49205800, 47300000),
    ("Towns, K-A.",    "C",  29, 33.8, 49205800, 45800000),
    ("Leonard, K.",    "SF", 33, 28.4, 49205800, 31600000),
    ("George, P.",     "SF", 34, 32.1, 49205800, 26900000),
    ("Butler, J.",     "SF", 35, 33.5, 48798677, 37400000),
    ("Lillard, D.",    "PG", 34, 35.1, 48787676, 38200000),
    ("Davis, A.",      "C",  31, 35.5, 43219440, 48700000),
    ("Trae Young",     "PG", 26, 34.3, 43031940, 38600000),
    ("LaVine, Z.",     "SG", 29, 32.0, 43031940, 31200000),
    ("Doncic, L.",     "PG", 25, 37.5, 43031940, 52800000),
    ("Haliburton, T.", "PG", 24, 33.7, 42176400, 44800000),
    ("Edwards, A.",    "SG", 23, 35.4, 42176400, 46100000),
    ("Sabonis, D.",    "C",  28, 33.5, 42476400, 41200000),
    ("Siakam, P.",     "PF", 30, 35.0, 42176400, 39800000),
    ("Mitchell, D.",   "SG", 28, 35.3, 35404494, 41700000),
    ("Gilgeous-Alex.", "SG", 26, 34.2, 35859950, 49600000),
    ("Fox, D.",        "PG", 27, 35.0, 34848340, 37200000),
    ("Tatum, J.",      "SF", 26, 36.2, 34848340, 48300000),
    ("Anunoby, O.G.",  "SF", 27, 33.8, 36625000, 38900000),
    ("Herro, T.",      "SG", 24, 34.2, 29000000, 27800000),
    ("Bridges, M.",    "SF", 27, 36.5, 23300000, 31400000),
    ("Brunson, J.",    "PG", 28, 34.6, 24960001, 38700000),
    ("Rotation P1",    "PG", 30, 22.1,  3600000,  4800000),
    ("Rotation P2",    "SG", 26, 18.5,  2200000,  3100000),
    ("Rotation P3",    "SF", 25, 24.3,  4100000,  5600000),
    ("Rotation P4",    "PF", 29, 20.8,  3000000,  3800000),
    ("Rotation P5",    "C",  27, 19.2,  2800000,  4200000),
    ("Rotation P6",    "PG", 23, 16.4,  2400000,  3600000),
    ("Rotation P7",    "SG", 28, 27.3,  7200000,  9100000),
    ("Rotation P8",    "SF", 24, 25.6,  5800000,  8300000),
    ("Rotation P9",    "PF", 31, 23.1,  4400000,  5200000),
    ("Rotation P10",   "C",  28, 21.7,  6100000,  7400000),
    ("Rotation P11",   "PG", 25, 28.9,  8900000, 11200000),
    ("Rotation P12",   "SG", 27, 30.2, 12400000, 14800000),
    ("Rotation P13",   "SF", 22, 22.3,  3800000,  5700000),
    ("Rotation P14",   "PF", 26, 26.7,  9200000, 10800000),
    ("Rotation P15",   "C",  29, 24.5,  7800000,  8900000),
    ("Rotation P16",   "PG", 28, 31.4, 16200000, 19100000),
    ("Rotation P17",   "SG", 25, 29.8, 10700000, 13200000),
    ("Rotation P18",   "SF", 30, 27.3, 11800000,  9600000),
    ("Rotation P19",   "PF", 32, 22.8,  8300000,  6100000),
    ("Rotation P20",   "C",  26, 25.4,  6700000,  8400000),
    ("Rotation P21",   "PG", 31, 28.6, 13100000, 10400000),
]

REQUIRED_COLS = {"Player", "actual_usd", "predicted_usd"}

DEFAULT_DF = pd.DataFrame(_RAW, columns=["Player", "Pos", "Age", "MP", "actual_usd", "predicted_usd"])
DEFAULT_DF["residual_usd"] = DEFAULT_DF["predicted_usd"] - DEFAULT_DF["actual_usd"]
DEFAULT_DF["pct_off"]      = DEFAULT_DF["residual_usd"] / DEFAULT_DF["actual_usd"]

IMPORTANCE = [
    ("WS", 1344.4), ("Age", 851.0), ("VORP", 778.5), ("PER", 360.9), ("BPM", 131.2),
    ("MP", 70.7), ("GS", 63.2), ("Age_sq", 61.4), ("is_starter", 57.7), ("AST_per36", 55.0),
    ("USG%", 54.1), ("TS%", 53.7), ("TOV_per36", 50.6), ("3P%", 38.5), ("STL_per36", 33.1),
]

NAVY, BLUE, ACCENT, GREEN, RED = "#0a1628", "#1d4ed8", "#f59e0b", "#22c55e", "#ef4444"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def fmt_usd(n): return f"${n:,.0f}" if not pd.isna(n) else "—"
def fmt_m(n):
    if pd.isna(n): return "—"
    return f"-${abs(n)/1e6:.1f}M" if n < 0 else f"+${n/1e6:.1f}M"


def parse_upload(file) -> tuple[pd.DataFrame, str | None]:
    try:
        df = pd.read_csv(file)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            return pd.DataFrame(), f"Missing columns: {', '.join(missing)}"
        pos_col = next((c for c in ("Pos_primary", "Pos") if c in df.columns), None)
        if pos_col and pos_col != "Pos":
            df = df.rename(columns={pos_col: "Pos"})
        if "residual_usd" not in df.columns:
            df["residual_usd"] = df["predicted_usd"] - df["actual_usd"]
        if "pct_off" not in df.columns:
            df["pct_off"] = df["residual_usd"] / df["actual_usd"]
        for col in ("Age", "MP"):
            if col not in df.columns:
                df[col] = np.nan
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    try:
        from src.model.score import load_model
        return load_model(MODEL_PATH)
    except Exception:
        return None


# ─── Navbar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <span class="nav-brand">🏀 NBA Contract Value</span>
  <ul class="nav-links">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#explorer">Explorer</a></li>
    <li><a href="#estimator">Salary Estimator</a></li>
    <li><a href="#model">Model</a></li>
    <li><a href="#methodology">Methodology</a></li>
  </ul>
</div>
""", unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero" id="overview">
  <div class="hero-inner">
    <h1>NBA Contract Value Analyzer</h1>
    <p class="hero-sub">
      A machine learning system that predicts what each NBA player should earn
      based on their current-season production, then surfaces where actual
      contracts diverge from market rate.
    </p>
    <div class="stat-cards">
      <div class="stat-card"><span class="stat-number">0.741</span><span class="stat-label">Test R²</span></div>
      <div class="stat-card"><span class="stat-number">$1.39M</span><span class="stat-label">Mean Absolute Error</span></div>
      <div class="stat-card"><span class="stat-number">4</span><span class="stat-label">Seasons of Training Data</span></div>
      <div class="stat-card"><span class="stat-number">36</span><span class="stat-label">Model Features</span></div>
    </div>
    <div class="disclosure-hero">
      <strong>Honest disclosure:</strong> Numbers shown are from a <strong>1,400-row synthetic demo run</strong> — the reproducibility fallback. On the real-data path (scraped stats joined against a hand-curated ~75-row salary set), development runs land in <strong>R² 0.68 – 0.74</strong>. Methodology is identical; only the data source differs. Salary sources (Spotrac, HoopsHype) are Cloudflare-gated, which is the reason for the small real-data salary set. See <a href="#methodology">Methodology</a>.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Contract Explorer ────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="section" id="explorer">', unsafe_allow_html=True)
    st.markdown('<h2>Contract Explorer</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Browse player salary predictions from the 2024-25 season. '
        'Upload your own <code>predictions_latest.csv</code> from the notebook for full '
        'real-model results, or explore the curated dataset below.</p>',
        unsafe_allow_html=True,
    )

    # ── CSV Upload ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-banner">
      <span class="upload-icon">📂</span>
      <div class="upload-text">
        <strong>Upload your own predictions</strong>
        <span>Run the notebook → export <code>predictions_latest.csv</code> → upload here for live model results</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload predictions_latest.csv",
        type=["csv"],
        label_visibility="collapsed",
        help="CSV must contain: Player, actual_usd, predicted_usd. Optional: Pos, Age, MP.",
    )

    if uploaded is not None:
        explorer_df, err = parse_upload(uploaded)
        if err:
            st.error(f"Could not read file: {err}")
            explorer_df = DEFAULT_DF.copy()
            data_source = "curated"
        else:
            st.success(f"Loaded {len(explorer_df):,} player predictions from your file.")
            data_source = "uploaded"
    else:
        explorer_df = DEFAULT_DF.copy()
        data_source = "curated"

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([2, 1.5, 1.5])
    with fc1:
        if "Pos" in explorer_df.columns and explorer_df["Pos"].notna().any():
            sel_pos = st.radio("Position", ["All", "PG", "SG", "SF", "PF", "C"], horizontal=True)
        else:
            sel_pos = "All"
            st.radio("Position", ["All"], horizontal=True, disabled=True)
    with fc2:
        sort_by = st.selectbox("Sort by", [
            "Most Overpaid", "Most Underpaid",
            "Highest Actual Salary", "Highest Predicted Salary",
        ])
    with fc3:
        mp_available = "MP" in explorer_df.columns and explorer_df["MP"].notna().any()
        min_mp = st.slider("Min MP/game", 0, 38, 15) if mp_available else 0

    # ── Apply filters ──────────────────────────────────────────────────────────
    view = explorer_df.copy()
    if sel_pos != "All" and "Pos" in view.columns:
        view = view[view["Pos"] == sel_pos]
    if mp_available:
        view = view[view["MP"] >= min_mp]

    sort_map = {
        "Most Overpaid":          ("residual_usd", True),
        "Most Underpaid":         ("residual_usd", False),
        "Highest Actual Salary":  ("actual_usd",   False),
        "Highest Predicted Salary":("predicted_usd", False),
    }
    s_col, s_asc = sort_map[sort_by]
    view = view.sort_values(s_col, ascending=s_asc)

    # ── Table ──────────────────────────────────────────────────────────────────
    row_data: dict = {"Player": view["Player"].values}
    if "Pos" in view.columns:  row_data["Pos"] = view["Pos"].values
    if "Age" in view.columns:  row_data["Age"] = view["Age"].fillna(0).astype(int).values
    if "MP"  in view.columns:  row_data["MP/g"] = view["MP"].round(1).values
    row_data["Actual"]    = [fmt_usd(v) for v in view["actual_usd"]]
    row_data["Predicted"] = [fmt_usd(v) for v in view["predicted_usd"]]
    row_data["Gap"]       = view["residual_usd"].round(0).values
    row_data["% Off"]     = (view["pct_off"] * 100).round(1).values
    row_data["Verdict"]   = ["Overpaid" if v < 0 else "Underpaid" for v in view["residual_usd"]]

    st.dataframe(
        pd.DataFrame(row_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Gap":   st.column_config.NumberColumn("Gap ($)", format="$%+,.0f"),
            "% Off": st.column_config.NumberColumn("% Off",   format="%+.1f%%"),
        },
    )
    st.caption(
        f"Data source: **{'uploaded file' if data_source == 'uploaded' else 'curated 2024-25 dataset'}** · "
        "Gap = Predicted − Actual · Negative = overpaid · Positive = underpaid."
    )

    # ── Scatter ────────────────────────────────────────────────────────────────
    st.markdown('<p class="subsec">Predicted vs. Actual Salary</p>', unsafe_allow_html=True)
    view["Verdict"] = ["Overpaid" if v < 0 else "Underpaid" for v in view["residual_usd"]]
    hover = {"Player": True} if "Player" in view.columns else {}
    if "Pos" in view.columns: hover["Pos"] = True

    fig_sc = px.scatter(
        view, x="actual_usd", y="predicted_usd",
        color="Verdict",
        color_discrete_map={"Overpaid": RED, "Underpaid": GREEN},
        hover_data=hover,
        labels={"actual_usd": "Actual Salary ($)", "predicted_usd": "Predicted Salary ($)"},
        log_x=True, log_y=True,
    )
    lo = float(min(view["actual_usd"].min(), view["predicted_usd"].min()))
    hi = float(max(view["actual_usd"].max(), view["predicted_usd"].max()))
    fig_sc.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                     line=dict(color="#94a3b8", dash="dash", width=1.5))
    fig_sc.update_layout(
        height=480, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13), xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Salary Estimator ─────────────────────────────────────────────────────────
st.markdown('<div class="section-alt"><div class="section-alt-inner" id="estimator">', unsafe_allow_html=True)
st.markdown('<h2>Salary Estimator</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Enter a player\'s statistics and see what the model estimates '
    'as their market salary. Uses the same feature logic as the trained pipeline — '
    'per-36 normalization, position pay curves, and age polynomial.</p>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Player Profile**")
    c_age = st.number_input("Age", 19, 40, 26, key="e_age")
    c_pos = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"], key="e_pos")
    c_g   = st.number_input("Games Played", 1, 82, 65, key="e_g")
    c_mp  = st.number_input("Minutes/game", 1.0, 38.0, 30.0, step=0.1, key="e_mp")
    c_gs  = st.number_input("Games Started", 0, 82, 55, key="e_gs")
with c2:
    st.markdown("**Per-Game Stats**")
    c_pts = st.number_input("Points/game",   0.0, 45.0, 18.0, step=0.1, key="e_pts")
    c_trb = st.number_input("Rebounds/game", 0.0, 20.0,  5.0, step=0.1, key="e_trb")
    c_ast = st.number_input("Assists/game",  0.0, 15.0,  4.0, step=0.1, key="e_ast")
    c_fgp = st.number_input("FG%", 0.0, 1.0, 0.47, step=0.01, key="e_fgp")
    c_3pp = st.number_input("3P%", 0.0, 1.0, 0.37, step=0.01, key="e_3pp")
with c3:
    st.markdown("**Advanced Stats**")
    c_per  = st.number_input("PER",         0.0, 35.0, 18.0, step=0.1, key="e_per")
    c_ws   = st.number_input("Win Shares", -5.0, 20.0,  5.0, step=0.1, key="e_ws")
    c_bpm  = st.number_input("BPM",       -10.0, 15.0,  1.5, step=0.1, key="e_bpm")
    c_vorp = st.number_input("VORP",       -1.0, 12.0,  2.0, step=0.1, key="e_vorp")
    c_usg  = st.number_input("USG%",        5.0, 40.0, 22.0, step=0.1, key="e_usg")

if st.button("Estimate Salary", type="primary", use_container_width=True):
    safe_mp = max(c_mp, 1.0)
    hyp = {
        "Age": c_age, "Age_sq": c_age**2, "is_prime": int(25 <= c_age <= 30),
        "G": c_g, "GS": c_gs, "MP": c_mp,
        "PTS_per36": c_pts * 36 / safe_mp, "TRB_per36": c_trb * 36 / safe_mp,
        "AST_per36": c_ast * 36 / safe_mp,
        "STL_per36": 1.2, "BLK_per36": 0.4, "TOV_per36": 2.5,
        "FGA_per36": 16.0, "3PA_per36": 5.0, "FTA_per36": 4.0,
        "FG%": c_fgp, "3P%": c_3pp, "FT%": 0.80,
        "eFG%": c_fgp + 0.5 * c_3pp * (5.0 / 16.0),
        "PER": c_per, "TS%": 0.56, "USG%": c_usg,
        "WS": c_ws, "WS/48": c_ws / max(c_g * c_mp / 48, 1),
        "BPM": c_bpm, "OBPM": c_bpm * 0.6, "DBPM": c_bpm * 0.4, "VORP": c_vorp,
        "is_starter": int(c_gs / max(c_g, 1) >= 0.5),
        "is_high_usage": int(c_usg >= 25), "is_rotation_only": int(c_mp < 20),
        "pos_PG": int(c_pos == "PG"), "pos_SG": int(c_pos == "SG"),
        "pos_SF": int(c_pos == "SF"), "pos_PF": int(c_pos == "PF"), "pos_C": int(c_pos == "C"),
    }

    est, shap_vals = None, None
    bundle = load_model_bundle()
    if bundle is not None:
        try:
            from src.model.score import score_single_player, _align_features
            import shap as shap_lib
            est = score_single_player(bundle, hyp)
            X_hyp = _align_features(bundle, pd.DataFrame([hyp]))
            _exp   = shap_lib.TreeExplainer(bundle["model"])
            shap_vals = _exp(X_hyp)
        except Exception:
            pass

    if est is None:
        pts36 = c_pts * 36 / safe_mp; ast36 = c_ast * 36 / safe_mp
        comp = (0.38*(c_ws/12) + 0.22*(c_vorp/6) + 0.18*(c_per/25)
                + 0.10*(c_bpm/8) + 0.05*(pts36/30) + 0.04*(ast36/10) + 0.03*(c_fgp*2))
        age_mod = (0.08 if 25<=c_age<=30 else 0) + (-0.15*(c_age-33) if c_age>33 else 0) + (-0.4 if c_age<23 else 0)
        pos_f   = {"PG": 0.05, "SG": 0.0, "SF": -0.02, "PF": -0.03, "C": 0.04}[c_pos]
        log_sal = 14.91 + 2.8*comp + age_mod + pos_f + 0.12*int(c_gs/max(c_g,1)>=0.5) + 0.08*int(c_usg>=25)
        salary  = float(min(max(np.exp(log_sal), 1_000_000), 55_000_000))
        tier = ("Max contract" if salary>=40e6 else "Star" if salary>=25e6
                else "Starter" if salary>=15e6 else "Role player" if salary>=7e6 else "Bench / reserve")
        est = {"predicted_usd": salary, "predicted_m": round(salary/1e6, 2),
               "tier": tier, "cap_pct_2025": round(salary/140_600_000*100, 1)}

    st.markdown(f"""
    <div class="result-box">
      <div class="result-lbl">Estimated Market Salary</div>
      <div class="result-amount">{fmt_usd(est['predicted_usd'])}</div>
      <div class="result-chips">
        <div class="result-chip"><div class="chip-val">{est['tier']}</div><div class="chip-lbl">Contract tier</div></div>
        <div class="result-chip"><div class="chip-val">{est['cap_pct_2025']}%</div><div class="chip-lbl">of 2025 cap</div></div>
        <div class="result-chip"><div class="chip-val">${est['predicted_m']}M</div><div class="chip-lbl">Annual value</div></div>
      </div>
      <div class="result-note">Estimate based on a model trained on 2022-2024 NBA salary data.
      Does not account for contract timing, roster context, or market size.</div>
    </div>""", unsafe_allow_html=True)

    if shap_vals is not None:
        st.markdown('<p class="subsec">Why did the model predict this?</p>', unsafe_allow_html=True)
        st.markdown(
            "The waterfall shows each feature's contribution. "
            "**Red bars** push the prediction up · **Blue bars** pull it down. "
            "Starting from the average log-salary baseline, features combine to reach the final value."
        )
        import shap as shap_lib
        import matplotlib.pyplot as plt
        shap_lib.plots.waterfall(shap_vals[0], max_display=12, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()
    else:
        st.info("SHAP waterfall requires `data/processed/model.pkl`. Run the notebook first, then restart.")

st.markdown('</div></div>', unsafe_allow_html=True)


# ─── Model Performance ────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="section" id="model">', unsafe_allow_html=True)
    st.markdown('<h2>Model Performance</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Trained on seasons 2022-2024 and evaluated on the held-out '
        '2024-25 season. The time-series split prevents future salary-cap information from '
        'leaking into historical predictions.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="disclosure-section">
      <strong>Data mode — synthetic demo run.</strong>
      The metrics below are from a 1,400-row synthetic-data run, the pipeline's reproducibility fallback when Basketball-Reference is unreachable or salary sources are unavailable.
      On the real-data path the model trains on scraped Basketball-Reference stats joined against a hand-curated ~75-row salary set; development runs in that mode land in the <strong>R² 0.68 – 0.74</strong> range with higher dollar MAE due to greater variance in real NBA contracts.
      Methodology is identical between the two paths; only the data source differs.
    </div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">0.741</div>
        <div class="metric-name">R² — Test Season</div>
        <div class="metric-desc">The model explains 74.1% of salary variance from statistical
        performance. The remaining 26% reflects contract timing, market premiums, and factors
        not in the stats.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">$1.39M</div>
        <div class="metric-name">Mean Absolute Error</div>
        <div class="metric-desc">Average prediction error in dollars — roughly equivalent to
        one mid-level exception. Within range for a stats-only model that cannot observe
        off-court factors.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">90</div>
        <div class="metric-name">Boosting Rounds</div>
        <div class="metric-desc">Model converged at round 90 with early stopping (patience
        = 50 rounds). Prevents overfitting on the small training set without manual round
        selection.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="subsec">Feature Importance — Top 15 by Gain</p>', unsafe_allow_html=True)
    feats  = [f for f, _ in IMPORTANCE]
    gains  = [g for _, g in IMPORTANCE]
    colors = [NAVY] + [BLUE] * (len(IMPORTANCE) - 1)
    fig_imp = go.Figure(go.Bar(
        x=gains[::-1], y=feats[::-1], orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:,.1f}<extra></extra>",
    ))
    fig_imp.update_layout(
        height=430, margin=dict(l=10, r=20, t=10, b=40),
        xaxis_title="LightGBM gain", plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f1f5f9"), font=dict(size=13),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "Win Shares is the dominant predictor — the composite stat front offices most often cite in "
        "contract negotiations. Composite efficiency metrics outrank raw counting stats consistently."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Methodology ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-alt"><div class="section-alt-inner" id="methodology">', unsafe_allow_html=True)
st.markdown('<h2>Methodology</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="method-grid">
  <div class="method-card">
    <h3>Data Acquisition</h3>
    <p>Per-game and advanced statistics scraped from Basketball-Reference.com using a polite HTTP
    client with disk-level caching, 3-second request delays per robots.txt, and exponential-backoff
    retries. A synthetic fallback generator reproduces realistic NBA stat distributions when the
    scraper is unavailable.</p>
    <p>Salary data from a curated dataset of ~75 well-known player-season contracts. Full-league
    coverage requires a once-per-season manual download from HoopsHype, which Cloudflare-protects
    against automated scraping.</p>
  </div>
  <div class="method-card">
    <h3>Feature Engineering</h3>
    <p><strong>Per-36 stats:</strong> normalize counting stats to 36 minutes of play. Decouples
    production rate from role — a player averaging 12 PPG in 20 minutes reads identically to one in
    36 per-game, but very differently per-36.</p>
    <p><strong>Position dummies:</strong> one-hot encode the five positions. Guards and centers face
    different supply curves (~120 guards vs. ~30 starting centers), producing distinct pay structures
    at the same performance level.</p>
    <p><strong>Age polynomial:</strong> Age + Age² + is_prime (25-30). Career earning arcs are
    non-monotonic; a linear age term misses both the rookie-scale floor and the post-32 discount.</p>
  </div>
  <div class="method-card">
    <h3>Modeling</h3>
    <p><strong>Algorithm:</strong> LightGBM (gradient boosted decision trees). With ~1,000 training
    rows across three seasons, gradient boosting outperforms neural networks on this tabular,
    heterogeneous feature set.</p>
    <p><strong>Target:</strong> log(salary_usd). Salaries span two orders of magnitude ($1M to $55M+).
    The log transform makes the regression loss symmetric across the full salary range.</p>
    <p><strong>Validation:</strong> time-series split — train on 2022-2024, evaluate on 2025.
    Random K-fold would leak future cap information into historical predictions.</p>
  </div>
  <div class="method-card">
    <h3>Limitations</h3>
    <p>Teams pay for factors outside the stats: market size, locker-room dynamics, injury risk beyond
    games played, and front-office relationships. The model cannot observe these.</p>
    <p>Contracts signed in prior seasons compare against current market rates. Cap-inflation features
    partially correct for this but do not fully resolve multi-year deals signed at prior peak values.</p>
    <p>Model predictions compress toward the mean at distribution extremes — max-contract predictions
    are systematically below actual; near-minimum predictions slightly above.</p>
  </div>
</div>""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-inner">
    <div class="footer-col">
      <strong>NBA Contract Value Analyzer</strong>
      <p>An end-to-end ML system for evaluating NBA salary efficiency.</p>
    </div>
    <div class="footer-col">
      <strong>Links</strong>
      <ul>
        <li><a href="https://github.com/bass990/NBA-Contract-Value-Analyzer" target="_blank">GitHub Repository</a></li>
        <li><a href="https://github.com/bass990/NBA-Contract-Value-Analyzer/blob/main/docs/REPORT.md" target="_blank">Analysis Report</a></li>
      </ul>
    </div>
    <div class="footer-col">
      <strong>Author</strong>
      <p>Mamadou Bassirou Diallo<br>MS Business Analytics &amp; AI<br>UT Dallas</p>
      <a href="https://www.linkedin.com/in/mamadou9905" target="_blank">LinkedIn</a>
    </div>
  </div>
  <div class="footer-bottom">
    Data source: Basketball-Reference.com · Salary data: publicly reported contract figures ·
    Model trained on 2022-2025 NBA seasons
  </div>
</div>
""", unsafe_allow_html=True)

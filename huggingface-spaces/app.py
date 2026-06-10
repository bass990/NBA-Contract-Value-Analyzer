"""Hugging Face Spaces entry point.

HF Spaces runs `streamlit run app.py` from the Space root. The real dashboard
lives at `src/app/dashboard.py` so the local-dev workflow (`make app` /
`streamlit run src/app/dashboard.py`) stays unchanged.

This wrapper executes the dashboard module in __main__ scope so Streamlit
processes its top-level calls exactly as it would when run directly.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DASHBOARD_PATH = ROOT / "src" / "app" / "dashboard.py"
if not DASHBOARD_PATH.exists():
    raise FileNotFoundError(
        f"Expected dashboard at {DASHBOARD_PATH}. "
        "Did the repo upload to the Space include the src/ tree?"
    )

runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")

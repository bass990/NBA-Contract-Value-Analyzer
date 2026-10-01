"""Contract, regression, adversarial and monitoring tests (no network).

* training is deterministic: a fresh `scripts/train.py` run reproduces the
  committed bundle's held-out metrics exactly;
* golden predictions: three fixed stat lines must score to the frozen
  salaries (any silent change to features, the pipeline or the booster fails);
* intervals and staleness on every prediction; drift PSI on batches;
* adversarial inputs the schema must reject or the model must survive.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import monitoring
from api.main import app
from src.model.score import load_model

GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden_predictions.json"
BUNDLE = load_model(PROJECT_ROOT / "data" / "processed" / "model.pkl")


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _star(player="Star", season=2025, **over) -> dict:
    d = {"Player": player, "season": season, "Age": 27, "Pos": "PG", "G": 70, "GS": 70, "MP": 35.0, "PTS": 28.5, "TRB": 5.5, "AST": 8.0,
         "STL": 1.3, "BLK": 0.4, "TOV": 3.1, "FGA": 20.5, "3PA": 9.0, "FTA": 7.0, "FG%": 0.48, "3P%": 0.38, "FT%": 0.89, "eFG%": 0.56,
         "PER": 24.0, "TS%": 0.61, "USG%": 32.0, "WS": 9.5, "WS/48": 0.18, "BPM": 6.0, "OBPM": 5.0, "DBPM": 1.0, "VORP": 4.5}
    d.update(over)
    return d


def _bench(player="Bench", season=2025, **over) -> dict:
    return _star(player, season, Age=24, Pos="SF", G=50, GS=5, MP=15.0, PTS=5.0, TRB=2.0, AST=1.0, STL=0.4, BLK=0.2, TOV=0.8, FGA=4.5,
                 **{"3PA": 1.5}, FTA=1.0, PER=9.0, **{"USG%": 14.0}, WS=0.8, **{"WS/48": 0.04}, BPM=-2.0, OBPM=-1.5, DBPM=-0.5, VORP=0.1, **over)


def _vet(player="Vet", season=2025) -> dict:
    return _star(player, season, Age=35, Pos="C", MP=24.0, PTS=12.0, TRB=8.0, AST=2.0, PER=16.0, **{"USG%": 20.0}, WS=4.0, BPM=1.0, VORP=1.2)


# ── reproducibility ─────────────────────────────────────────────────────────

def test_bundle_carries_v2_fields():
    for key in ("residual_quantiles_log", "reference", "training_seasons", "validation_season", "test_season", "trained_at", "top_features"):
        assert key in BUNDLE, key
    assert BUNDLE["validation_season"] < BUNDLE["test_season"]
    assert max(BUNDLE["training_seasons"]) < BUNDLE["validation_season"], "early stopping must not see the test season"


def test_training_is_deterministic(tmp_path):
    """Retrain into a temp path and compare the held-out metrics with the committed bundle."""
    out, rep = tmp_path / "m.pkl", tmp_path / "r.json"
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--out", str(out), "--report", str(rep)],
                   check=True, capture_output=True, cwd=PROJECT_ROOT, timeout=600)
    fresh = load_model(out)
    assert fresh["test_r2"] == pytest.approx(BUNDLE["test_r2"], abs=1e-9)
    assert fresh["test_mae_usd"] == pytest.approx(BUNDLE["test_mae_usd"], abs=1.0)
    assert fresh["model"].num_trees() == BUNDLE["model"].num_trees()


def test_golden_predictions(client):
    inputs = [_star(), _bench(), _vet()]
    preds = [p["predicted_usd"] for p in client.post("/predict-batch", json={"players": inputs}).json()["predictions"]]
    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_text(json.dumps({"note": "frozen by tests/test_contract.py::test_golden_predictions on first run", "predicted_usd": preds}, indent=1))
        pytest.skip("golden file created")
    golden = json.loads(GOLDEN_PATH.read_text())["predicted_usd"]
    for got, want in zip(preds, golden, strict=True):
        assert abs(got - want) <= 1, f"prediction drifted: {got} vs golden {want}"


# ── intervals, staleness, ordering ──────────────────────────────────────────

def test_prediction_has_interval_and_stale_flag(client):
    p = client.post("/predict", json=_star(season=2025)).json()["predictions"][0]
    assert p["interval_80_low_usd"] < p["predicted_usd"] < p["interval_80_high_usd"]
    assert p["stale"] is False
    q = BUNDLE["residual_quantiles_log"]
    assert p["interval_80_low_usd"] == pytest.approx(p["predicted_usd"] * np.exp(q["q10"]), rel=1e-3)
    far = client.post("/predict", json=_star(season=2030)).json()["predictions"][0]
    assert far["stale"] is True


def test_star_vet_bench_ordering(client):
    preds = client.post("/predict-batch", json={"players": [_star(), _vet(), _bench()]}).json()["predictions"]
    assert preds[0]["predicted_usd"] > preds[1]["predicted_usd"] > preds[2]["predicted_usd"]
    assert [p["Player"] for p in preds] == ["Star", "Vet", "Bench"]


def test_model_info_reports_evaluation_and_leak_fix(client):
    body = client.get("/model-info").json()
    assert body["test_season"] == 2025 and body["validation_season"] == 2024 and body["training_seasons"] == [2022, 2023]
    assert 0.6 < body["test_r2"] < 0.8 and body["interval_80_coverage_test"] > 0.6
    assert "synthetic" in body["honest_disclosure"].lower()
    assert body["held_out_evaluation"]["r2_log"] == pytest.approx(body["test_r2"], abs=1e-3)
    assert any(t["tier"] == "Max contract" for t in body["error_by_salary_tier"])
    assert body["stale_after_season"] == 2025


def test_headers(client):
    r = client.get("/health", headers={"X-Request-ID": "req-1"})
    assert r.headers["X-Request-ID"] == "req-1" and "X-Process-Time-Ms" in r.headers
    assert r.json()["drift_reference_loaded"] is True


# ── adversarial inputs ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    {"MP": 0.0},                      # per-36 division by zero -> NaN path
    {"G": 0, "GS": 0},                # is_starter division by zero
    {"Age": 50}, {"Age": 18},
    {"PTS": 60.0, "USG%": 100.0, "PER": 60.0, "WS": 25.0, "VORP": 15.0, "BPM": 20.0},  # beyond anything in training
    {"FG%": 0.0, "3P%": 0.0, "FT%": 0.0, "eFG%": 0.0, "TS%": 0.0},
    {"Pos": "C-PF"},
])
def test_predict_survives_edge_inputs(client, bad):
    r = client.post("/predict", json=_star(**bad))
    assert r.status_code == 200, r.text
    p = r.json()["predictions"][0]
    assert p["predicted_usd"] > 0 and np.isfinite(p["predicted_usd"])


@pytest.mark.parametrize("bad", [{"MP": 49.0}, {"G": 83}, {"FG%": 1.2}, {"season": 1900}, {"Player": ""}, {"Age": "old"}, {"USG%": -1}])
def test_predict_rejects_out_of_range(client, bad):
    assert client.post("/predict", json=_star(**bad)).status_code == 422


def test_batch_limit(client):
    r = client.post("/predict-batch", json={"players": [_star(player=f"p{i}") for i in range(1001)]})
    assert r.status_code == 422


# ── drift ───────────────────────────────────────────────────────────────────

def test_psi_math():
    ref = np.full(10, 0.1)
    assert monitoring.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert monitoring.psi(ref, np.array([0.6, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])) > 0.25


def test_drift_endpoint_flags_shifted_batch(client):
    rng = np.random.default_rng(0)
    normal = []
    for i in range(80):
        s = rng.gamma(2.0, 1.0)
        normal.append(_star(player=f"n{i}", Age=int(rng.integers(20, 36)), MP=float(np.clip(15 + s * 3, 5, 38)), PTS=float(5 + s * 3),
                            PER=float(8 + s * 2.5), **{"USG%": float(15 + s * 2)}, WS=float(s * 1.5), BPM=float(-3 + s * 1.5), VORP=float(s * 0.8)))
    body = client.post("/drift", json={"players": normal}).json()
    assert body["n_rows"] == 80 and body["features"]
    superstars = [_star(player=f"s{i}", PTS=35.0, PER=30.0, WS=14.0, VORP=8.0, BPM=10.0, **{"USG%": 38.0}) for i in range(80)]
    shifted = client.post("/drift", json={"players": superstars}).json()
    assert shifted["n_shifted"] >= body["n_shifted"]  # a batch of clones is already far from the reference
    assert shifted["features"]["PER"]["status"] == "shifted" and shifted["prediction_psi"]["status"] == "shifted"


def test_drift_refuses_small_batches_and_batch_attaches_summary(client):
    assert client.post("/drift", json={"players": [_star()]}).status_code == 422
    small = client.post("/predict-batch", json={"players": [_star(), _bench()]}).json()
    assert small["drift"] is None
    big = client.post("/predict-batch", json={"players": [_star(player=f"p{i}") for i in range(60)]}).json()
    assert big["drift"] is not None and big["drift"]["n_rows"] == 60

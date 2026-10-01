"""End-to-end tests for the NBA contract-value FastAPI service.

These tests boot the real lifespan handler (which loads the live model.pkl)
and exercise every endpoint through the FastAPI TestClient. No mocking of the
model — the pickle is small, the predict path is fast, and the contract under
test IS that the live pipeline + live model produce a sane response shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _star_guard(player: str = "Test Guard", season: int = 2024) -> dict:
    """A plausible all-star guard line — used to check the response shape, not
    to validate a specific predicted salary. The honest-disclosure caveat
    (synthetic-data training) means absolute predictions are illustrative."""
    return {
        "Player": player,
        "season": season,
        "Age": 27,
        "Pos": "PG",
        "G": 70,
        "GS": 70,
        "MP": 35.0,
        "PTS": 28.5,
        "TRB": 5.5,
        "AST": 8.0,
        "STL": 1.3,
        "BLK": 0.4,
        "TOV": 3.1,
        "FGA": 20.5,
        "3PA": 9.0,
        "FTA": 7.0,
        "FG%": 0.480,
        "3P%": 0.380,
        "FT%": 0.890,
        "eFG%": 0.560,
        "PER": 24.0,
        "TS%": 0.610,
        "USG%": 32.0,
        "WS": 9.5,
        "WS/48": 0.180,
        "BPM": 6.0,
        "OBPM": 5.0,
        "DBPM": 1.0,
        "VORP": 4.5,
    }


def _bench_player() -> dict:
    body = _star_guard(player="Bench Test", season=2024)
    body.update({
        "Age": 24, "Pos": "SF", "G": 50, "GS": 5, "MP": 15.0,
        "PTS": 5.0, "TRB": 2.0, "AST": 1.0, "STL": 0.4, "BLK": 0.2,
        "TOV": 0.8, "FGA": 4.5, "3PA": 1.5, "FTA": 1.0,
        "PER": 9.0, "USG%": 14.0, "WS": 0.8, "WS/48": 0.040,
        "BPM": -2.0, "OBPM": -1.5, "DBPM": -0.5, "VORP": 0.1,
    })
    return body


def test_health_after_startup(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_reports_synthetic_data_caveat(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == len(body["feature_names"]) == 36
    # The model-card numbers must match the trained bundle.
    assert 0.70 < body["test_r2"] < 0.78
    assert body["test_mae_usd"] > 1_000_000
    # The honest-disclosure caveat — synthetic-data framing must travel with the API.
    assert "synthetic" in body["honest_disclosure"].lower()


def test_predict_single_player_returns_valid_shape(client: TestClient) -> None:
    r = client.post("/predict", json=_star_guard())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["Player"] == "Test Guard"
    assert pred["season"] == 2024
    assert pred["predicted_usd"] > 0
    assert pred["predicted_m"] == round(pred["predicted_usd"] / 1_000_000, 2)
    assert pred["tier"] in {"Max contract", "Star", "Starter", "Role player", "Bench / reserve"}
    assert 0.0 <= pred["cap_pct_2025"] <= 200.0  # could exceed 100% pre-cap-rule predictions


def test_predict_star_outranks_bench_predicted_salary(client: TestClient) -> None:
    """Even on a model trained on synthetic data, a star line should out-predict
    a bench line — this verifies the per-36 + role-flag pipeline plumbed through."""
    star = client.post("/predict", json=_star_guard()).json()["predictions"][0]
    bench = client.post("/predict", json=_bench_player()).json()["predictions"][0]
    assert star["predicted_usd"] > bench["predicted_usd"]


def test_predict_batch_preserves_order(client: TestClient) -> None:
    players = [
        _star_guard(player=f"Player {i}", season=2024) for i in range(3)
    ]
    r = client.post("/predict-batch", json={"players": players})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [p["Player"] for p in body["predictions"]] == ["Player 0", "Player 1", "Player 2"]


def test_predict_rejects_negative_minutes(client: TestClient) -> None:
    bad = _star_guard()
    bad["MP"] = -5.0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_unknown_position(client: TestClient) -> None:
    bad = _star_guard()
    bad["Pos"] = "QB"  # not a basketball position
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_extra_field(client: TestClient) -> None:
    bad = _star_guard()
    bad["secret_clutch_factor"] = 99.9
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch_rejects_empty(client: TestClient) -> None:
    r = client.post("/predict-batch", json={"players": []})
    assert r.status_code == 422

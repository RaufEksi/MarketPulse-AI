"""
Integration tests for FastAPI REST endpoints.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "marketpulse_api_requests_total" in response.text


def test_backtest_endpoint():
    payload = {
        "symbol": "SPY",
        "spike_threshold": 0.65,
        "hedge_reduction_factor": 0.2,
        "initial_capital": 100000.0,
    }
    response = client.post("/backtest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "strategy_metrics" in data
    assert "strategy_equity" in data


def test_explain_endpoint():
    payload = {
        "prediction_id": "mp-test-123",
        "symbol": "SPY",
        "top_k_features": 3,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_decomposition" in data
    assert len(data["top_features"]) == 3

"""
Integration tests for FastAPI REST endpoints.
"""

import numpy as np
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


def test_predict_endpoint(sample_ohlcv_df):
    bars_list = []
    for _, row in sample_ohlcv_df.head(25).iterrows():
        bars_list.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "vwap": float(row["vwap"]),
                "trade_count": int(row["trade_count"]),
            }
        )

    payload = {
        "symbol": "SPY",
        "ohlcv_bars": bars_list,
        "recent_texts": [
            {
                "timestamp": bars_list[-1]["timestamp"],
                "headline": "Fed signals steady rate path amid economic expansion.",
                "source": "news",
            }
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "volatility_spike_probability" in data
    assert "risk_level" in data
    assert 0.0 <= data["volatility_spike_probability"] <= 1.0


def test_explain_endpoint_with_custom_bars(sample_ohlcv_df):
    bars_list = []
    for _, row in sample_ohlcv_df.head(25).iterrows():
        bars_list.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "vwap": float(row["vwap"]),
                "trade_count": int(row["trade_count"]),
            }
        )

    payload = {
        "prediction_id": "mp-custom-456",
        "symbol": "SPY",
        "top_k_features": 4,
        "ohlcv_bars": bars_list,
        "recent_texts": [
            {
                "timestamp": bars_list[-1]["timestamp"],
                "headline": "Tech earnings surprise to the upside with massive cloud growth.",
                "source": "news",
            }
        ],
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_decomposition" in data
    assert len(data["top_features"]) <= 4
    assert "news_sentiment_pct" in data["risk_decomposition"]
    assert "technical_indicators_pct" in data["risk_decomposition"]


def test_predict_serves_trained_checkpoint(sample_ohlcv_df, tmp_path, monkeypatch):
    import torch

    from src.api import model_registry
    from src.config.settings import get_settings
    from src.models.checkpoint import save_checkpoint
    from src.models.hybrid_network import MarketPulseNet

    # Arrange: a checkpoint whose classifier bias forces a near-certain spike
    config = {"ts_input_dim": 16, "text_input_dim": 768, "hidden_dim": 32, "num_heads": 4}
    model = MarketPulseNet(**config)
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.fill_(10.0)
    ckpt = tmp_path / "trained.pt"
    save_checkpoint(
        model,
        ckpt,
        model_config=config,
        feature_columns=model_registry.MODEL_FEATURE_COLUMNS,
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_std=np.ones(16, dtype=np.float32),
        metadata={"text_embedder": "hash-fallback"},
    )
    monkeypatch.setattr(get_settings().model, "checkpoint_path", str(ckpt))
    model_registry.reset_registry()

    bars = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in sample_ohlcv_df.head(25).iterrows()
    ]

    try:
        # Act
        response = client.post("/predict", json={"symbol": "SPY", "ohlcv_bars": bars})

        # Assert
        assert response.status_code == 200
        assert response.json()["volatility_spike_probability"] > 0.99
        assert response.json()["risk_level"] == "CRITICAL_VOLATILITY"
        assert model_registry.get_model_bundle().metadata["source"] == str(ckpt)
    finally:
        model_registry.reset_registry()


def test_predict_without_checkpoint_uses_untrained_model(tmp_path, monkeypatch):
    from src.api import model_registry
    from src.config.settings import get_settings

    monkeypatch.setattr(get_settings().model, "checkpoint_path", str(tmp_path / "none.pt"))
    model_registry.reset_registry()
    try:
        bundle = model_registry.get_model_bundle()
        assert bundle.metadata["source"] == model_registry.UNTRAINED_MODEL_SOURCE
    finally:
        model_registry.reset_registry()


def test_market_bars_endpoint():
    response = client.get("/market/bars", params={"symbol": "spy", "limit": 30})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "SPY"
    assert len(data["bars"]) == 30
    first = data["bars"][0]
    for key in ("timestamp", "open", "high", "low", "close", "volume", "atr_14", "rsi_14"):
        assert key in first
    timestamps = [b["timestamp"] for b in data["bars"]]
    assert timestamps == sorted(timestamps)


def test_market_bars_rejects_too_few_bars():
    response = client.get("/market/bars", params={"symbol": "SPY", "limit": 5})
    assert response.status_code == 422


def test_market_texts_endpoint():
    response = client.get("/market/texts", params={"symbol": "NVDA", "limit": 7})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert 0 < len(data["texts"]) <= 7
    timestamps = [t["timestamp"] for t in data["texts"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_market_bars_feed_predict_endpoint():
    bars = client.get("/market/bars", params={"symbol": "SPY", "limit": 78}).json()["bars"]
    payload = {
        "symbol": "SPY",
        "ohlcv_bars": [
            {k: b[k] for k in ("timestamp", "open", "high", "low", "close", "volume")} for b in bars
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

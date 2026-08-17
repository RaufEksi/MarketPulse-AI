# MarketPulse AI — REST API Reference

MarketPulse AI exposes high-performance asynchronous REST endpoints built on FastAPI.

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict short-term volatility spike probability for a given ticker and state. |
| `POST` | `/explain` | Obtain SHAP feature attributions and risk factor decomposition. |
| `POST` | `/backtest` | Execute a historical strategy simulation against market benchmarks. |
| `GET` | `/health` | System health, readiness, and data pipeline status. |
| `GET` | `/metrics` | Prometheus metrics exporter. |

---

## Detailed Endpoint Specifications

### 1. `POST /predict`

**Request Body (`PredictRequest`):**
```json
{
  "symbol": "SPY",
  "ohlcv_bars": [
    {
      "timestamp": "2026-08-17T09:30:00Z",
      "open": 550.25,
      "high": 551.00,
      "low": 549.80,
      "close": 550.80,
      "volume": 1250000
    }
  ],
  "recent_texts": [
    {
      "timestamp": "2026-08-17T09:25:00Z",
      "headline": "Fed signals unexpected rate hike in upcoming FOMC meeting",
      "source": "newsapi"
    }
  ]
}
```

**Response Body (`PredictResponse`):**
```json
{
  "prediction_id": "mp-98f2b34a-912c",
  "symbol": "SPY",
  "timestamp": "2026-08-17T09:30:00Z",
  "volatility_spike_probability": 0.842,
  "risk_level": "CRITICAL_VOLATILITY",
  "confidence_interval": {
    "lower": 0.79,
    "upper": 0.89
  },
  "inference_latency_ms": 18.4
}
```

---

### 2. `POST /explain`

**Response Body (`ExplainResponse`):**
```json
{
  "prediction_id": "mp-98f2b34a-912c",
  "symbol": "SPY",
  "risk_decomposition": {
    "breaking_news_sentiment": 0.65,
    "rsi_divergence": 0.20,
    "volume_surge": 0.15
  },
  "top_features": [
    {"feature": "finbert_sentiment_dim_42", "shap_value": 0.34},
    {"feature": "rolling_volatility_12", "shap_value": 0.21},
    {"feature": "rsi_14", "shap_value": -0.12}
  ]
}
```

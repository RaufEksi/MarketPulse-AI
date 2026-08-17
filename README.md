# MarketPulse AI

**Multi-Modal Financial Volatility & Market Regime Shift Prediction System**

MarketPulse AI is an end-to-end explainable hybrid deep learning system that predicts short-term volatility spikes and market regime breaks by fusing real-time market data with financial sentiment signals.

## Core Objective

Instead of predicting directional movement (up/down), MarketPulse AI focuses on **volatility forecasting**: predicting whether the next 30 minutes will experience a significant volatility spike (≥15% ATR increase) by combining:

- **Time Series Analysis**: 5-minute OHLCV bars from Alpaca Markets
- **Sentiment Analysis**: Financial news & social media (Reddit, GDELT)
- **Deep Learning**: Bi-LSTM/TCN encoders + Multi-head Cross-Attention Fusion
- **Explainability**: SHAP, Integrated Gradients, Attention Visualization

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- Alpaca Markets API key (get [here](https://alpaca.markets/))
- Reddit PRAW credentials (get [here](https://www.reddit.com/prefs/apps))

### Local Setup

```bash
# Clone repository
git clone https://github.com/RaufEksi/MarketPulse-AI.git
cd MarketPulse-AI

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Setup with make
make setup

# Run tests
make test

# Start local services (API + Dashboard)
make serve
```

### Docker Setup

```bash
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Languages** | Python 3.10+ |
| **Time Series** | PyTorch, Bi-LSTM, TCN |
| **NLP** | HuggingFace (FinBERT, RoBERTa) |
| **Data** | Pandas, NumPy, Scikit-learn |
| **XAI** | SHAP, Integrated Gradients |
| **API** | FastAPI, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **DevOps** | Docker, Kubernetes, Terraform |
| **Monitoring** | Prometheus, Grafana, AlertManager |

## Project Structure

```
MarketPulse-AI/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   ├── data_engine/              # API connectors & raw data ingestion
│   ├── feature_engineering/      # Feature extraction & preprocessing
│   ├── data_alignment/           # Temporal alignment & labeling
│   ├── models/                   # Deep learning architectures & training
│   ├── xai_explainer/            # SHAP, Integrated Gradients, attention viz
│   ├── api/                      # FastAPI backend
│   ├── dashboard/                # Streamlit frontend
│   └── utils/                    # Helpers, decorators, exceptions
├── tests/                        # Unit & integration tests
├── docs/                         # Comprehensive documentation
├── deploy/                       # Docker, Kubernetes, Terraform configs
├── scripts/                      # Standalone utility scripts
├── notebooks/                    # Exploratory analysis & prototyping
├── config/                       # Environment-specific configs
├── data/                         # Raw & processed data (not versioned)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # Local dev stack
├── Makefile                      # Common dev tasks
└── README.md                     # This file
```

## Architecture Overview

```
Raw Data (Alpaca, Reddit, GDELT)
         ↓
   Data Ingestion & Validation
         ↓
   Feature Engineering (ATR, RSI, FinBERT embeddings)
         ↓
   Exponential Decay Temporal Alignment
         ↓
   Hybrid Model (Bi-LSTM + Text Encoder + Attention Fusion)
         ↓
   Explainability (SHAP + Integrated Gradients)
         ↓
   FastAPI Backend (/predict, /explain, /backtest)
         ↓
   Streamlit Dashboard
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Key Concepts

### Target Variable: ATR Volatility Spike

A **spike** occurs at bar `t` if:

```
max(ATR[t+1:t+6]) ≥ ATR[t] × 1.15
```

In English: "Does ATR increase by ≥15% in the next 6 bars (30 minutes)?"

### Temporal Alignment: Exponential Decay Forward-Fill

Fuse irregular text events with regular 5-min bars using:

```
S(t) = Σ [ S_i × e^(-λ × Δt) ]
```

Where `λ` controls the decay rate (default: 0.5 per hour). Recent events have strong influence; older events fade.

### Model Architecture: Hybrid Fusion

1. **Time Series Encoder**: Bi-LSTM or TCN on price features
2. **Text Encoder**: FinBERT sentiment embeddings
3. **Fusion Layer**: Multi-head Cross-Attention
4. **Classification Head**: Binary sigmoid (spike or no spike)

## API Endpoints

### `/predict` (POST)
Real-time volatility spike prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "ohlcv_bars": [...], "recent_sentiment": [...]}'
```

Response:
```json
{
  "prediction_id": "abc-123",
  "prediction": 0.87,
  "class": "spike_likely",
  "confidence_interval": {"lower": 0.82, "upper": 0.92},
  "latency_ms": 42
}
```

### `/explain` (POST)
SHAP-based feature importance for a prediction.

### `/backtest` (GET)
Historical strategy performance analysis.

### `/health` (GET)
System status & data freshness.

For complete API documentation, visit: http://localhost:8000/docs (Swagger UI)

## Dashboard Pages

- **Real-Time Monitor**: Live predictions & confidence bands
- **Explainability Explorer**: SHAP waterfall, attention heatmaps
- **Backtesting Engine**: Strategy performance analysis
- **Data Explorer**: OHLCV visualization, sentiment trends
- **System Health**: API latency, model drift, data pipeline status

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models/test_architectures.py -v
```

Target coverage: ≥80%

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Model Design](docs/MODEL_DESIGN.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Contributing](CONTRIBUTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, PR process, and code standards.

## License

MIT License - See [LICENSE](LICENSE) file.

## Project Lead

**Role**: Senior FinTech AI Architect & Lead MLOps Engineer

## References

Key papers & resources:
- LSTM Time Series Forecasting (Hochreiter & Schmidhuber)
- Temporal Convolutional Networks (Bai et al.)
- FinBERT: Financial Language Models (Huang et al.)
- SHAP: Explainable AI (Lundberg & Lee)
- Integrated Gradients (Sundararajan et al.)

---

**Status**: In Active Development  
**Last Updated**: 2026-07-25

# MarketPulse AI - Granular Issue & Task Catalog 📋

This document catalogues all 33 granular project issues organized across 8 Phases for **MarketPulse AI: Multi-Modal Financial Volatility & Market Regime Shift Prediction System**.

---

## 🏗️ PHASE 1: Project Setup, Infrastructure & Environment

#### **Issue #1-1**
**Title:** `[Infra] Configuration Management Engine with Pydantic & YAML`  
**Labels:** `infrastructure`, `backend`  
**Amaç & Kapsam:** Implement centralized environment configuration via Pydantic `BaseSettings` supporting YAML (`config/default.yaml`) and `.env` hierarchies.  
**Kabul Kriterleri:**
- [ ] `src/config/settings.py` with `Settings` class loading `config/default.yaml` and `.env`
- [ ] Support type-safe overrides for data, NLP, model, training, and API parameters
- [ ] Unit tests in `tests/test_config.py`  
**Alt Görevler:**
- [ ] Add dynamic environment switching (development, production)
- [ ] Implement settings caching with `@lru_cache`
- [ ] Add secret masking for sensitive API tokens

#### **Issue #1-2**
**Title:** `[Infra] Structured JSON Logging & Custom Domain Exceptions`  
**Labels:** `infrastructure`, `backend`  
**Amaç & Kapsam:** Create structured JSON logging engine and domain-specific exception hierarchy for standardized system errors.  
**Kabul Kriterleri:**
- [ ] `src/utils/logger.py` with `JSONFormatter` producing ISO8601 timestamps and execution metadata
- [ ] `src/utils/exceptions.py` with `MarketPulseException`, `DataIngestionError`, `ModelInferenceError`
- [ ] Zero unhandled tracebacks in production API logs

#### **Issue #1-3**
**Title:** `[DevOps] Multi-Stage Dockerfile & Local docker-compose Stack`  
**Labels:** `devops`, `infrastructure`  
**Amaç & Kapsam:** Build an optimized multi-stage `Dockerfile` and `docker-compose.yml` orchestrating FastAPI backend, Streamlit dashboard, and Prometheus monitoring.  
**Kabul Kriterleri:**
- [ ] Multi-stage Dockerfile (< 1.5GB image)
- [ ] `docker-compose.yml` with healthchecks and isolated bridge network
- [ ] Volume persistence for `/data` Parquet storage

#### **Issue #1-4**
**Title:** `[CI/CD] GitHub Actions Automated Testing, Linting & Security Gateways`  
**Labels:** `devops`, `infrastructure`  
**Amaç & Kapsam:** Establish automated CI/CD pipeline enforcing Black, isort, Flake8, Bandit security scanning, and pytest coverage across Python 3.10 and 3.11.  
**Kabul Kriterleri:**
- [ ] `.github/workflows/ci.yml` matrix pipeline
- [ ] Security vulnerability scanning with Bandit
- [ ] Code coverage threshold verification (>=80%)

---

## 📡 PHASE 2: Data Ingestion & Storage Lake

#### **Issue #2-1**
**Title:** `[Data] Alpaca Markets 5-Minute OHLCV Bar Collector`  
**Labels:** `data-engine`, `backend`  
**Amaç & Kapsam:** Ingest real-time and historical 5-minute OHLCV intraday market bars from Alpaca Markets Data API v2.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/alpaca_connector.py` with `AlpacaDataCollector`
- [ ] Automatic pagination and rate-limit backoff handling
- [ ] Returns validated DataFrame: `[timestamp, open, high, low, close, volume, vwap, trade_count]`
- [ ] Offline synthetic fallback generator

#### **Issue #2-2**
**Title:** `[Data] Yahoo Finance Fallback Connector (yfinance)`  
**Labels:** `data-engine`, `backend`  
**Amaç & Kapsam:** Provide seamless fallback mechanism using Yahoo Finance when Alpaca API credentials or quotas are unavailable.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/yfinance_connector.py` with `YFinanceDataCollector`
- [ ] Automatic column schema harmonization

#### **Issue #2-3**
**Title:** `[Data] Reddit PRAW Financial Sentiment Scraper`  
**Labels:** `data-engine`, `nlp`  
**Amaç & Kapsam:** Scrape ticker-tagged discussions from financial subreddits (`r/wallstreetbets`, `r/stocks`, `r/investing`) using PRAW.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/reddit_collector.py` with `RedditCollector`
- [ ] Filter by target symbols (`SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `TSLA`)
- [ ] Extracts title, score, comment count, and normalized UTC timestamp

#### **Issue #2-4**
**Title:** `[Data] NewsAPI & GDELT Financial Headline Ingestion Service`  
**Labels:** `data-engine`, `nlp`  
**Amaç & Kapsam:** Collect breaking institutional news headlines and regulatory alerts from NewsAPI / GDELT.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/news_collector.py` with `NewsCollector`
- [ ] Headline deduplication by URL and content hash
- [ ] Graceful quota limit handling

#### **Issue #2-5**
**Title:** `[Data Storage] Partitioned Parquet Data Lake & Local Storage Manager`  
**Labels:** `data-engine`, `infrastructure`  
**Amaç & Kapsam:** Implement Parquet storage engine partitioned by ticker symbol, source, and date for raw and processed datasets.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/storage_manager.py` with `StorageManager`
- [ ] Partition structure: `data/raw/{source}/symbol={symbol}/data.parquet`
- [ ] Sub-100ms load times for 100k rows with PyArrow

#### **Issue #2-6**
**Title:** `[Data Storage] Raw Data Organization & Retention Policy`  
**Labels:** `data-engine`, `infrastructure`  
**Amaç & Kapsam:** Organize raw data storage directories and establish automated retention/archival policies (e.g. 12 months rolling window).  
**Kabul Kriterleri:**
- [ ] Manifest file metadata `{date}_manifest.json`
- [ ] Retention cleanup utility
- [ ] Compression optimization (Snappy/Gzip)

---

## ⚙️ PHASE 3: Feature Engineering & Temporal Alignment

#### **Issue #3-1**
**Title:** `[Features] Time Series Feature Extraction (ATR, RSI, MACD, Bollinger Bands)`  
**Labels:** `feature-engineering`, `data-science`  
**Amaç & Kapsam:** Calculate comprehensive technical momentum and volatility indicators on 5-minute bars without lookahead bias.  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/technical_indicators.py`
- [ ] Implements ATR(14), RSI(14), MACD(12,26,9), Bollinger Bands(20,2), Rolling Volatility (12, 36, 78 bars)
- [ ] Unit tests in `tests/test_feature_engineering.py`

#### **Issue #3-2**
**Title:** `[Features] Volume & Micro-Structure Features (VWAP Divergence, Volume Ratio)`  
**Labels:** `feature-engineering`, `data-science`  
**Amaç & Kapsam:** Extract micro-structural volume features, volume moving average ratios, and VWAP deviation scores.  
**Kabul Kriterleri:**
- [ ] Volume to 20-period moving average ratio
- [ ] Log-return calculation
- [ ] Handling market opening/closing spikes

#### **Issue #3-3**
**Title:** `[Labels] ATR Volatility Spike Ground-Truth Labeling Generator`  
**Labels:** `feature-engineering`, `data-science`  
**Amaç & Kapsam:** Generate binary classification labels defined as: max(ATR[t+1:t+6]) >= 1.15 * ATR[t] (15% ATR increase over next 30 minutes).  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/labeler.py` with `VolatilityLabeler`
- [ ] Strict exclusion of lookahead leak during validation
- [ ] Generates 'atr_spike_target' and 'volatility_expansion_ratio'

#### **Issue #3-4**
**Title:** `[NLP] Financial Text Cleaning, Normalization & Tokenization Pipeline`  
**Labels:** `nlp`, `feature-engineering`  
**Amaç & Kapsam:** Clean and normalize unstructured financial texts (URL removal, HTML stripping, cashtag normalizations like $AAPL -> AAPL, length filtering).  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/text_preprocessor.py` with `TextPreprocessor`
- [ ] Length constraints (10 to 1000 characters)
- [ ] Unit tests verifying regex cleaning

#### **Issue #3-5**
**Title:** `[NLP] FinBERT Sentiment 768-D Embedding Generation & Cache Engine`  
**Labels:** `nlp`, `deep-learning`  
**Amaç & Kapsam:** Extract 768-dimensional sentence embeddings using pre-trained FinBERT (`ProsusAI/finbert`) with GPU batching and disk caching.  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/sentiment_embedder.py` with `FinBERTEmbedder`
- [ ] Extract [CLS] token embeddings [768-D]
- [ ] Batch processing (32 texts) with CPU/GPU device selection

#### **Issue #3-6**
**Title:** `[Alignment] Exponential Decay Forward-Fill Temporal Alignment Engine`  
**Labels:** `feature-engineering`, `backend`  
**Amaç & Kapsam:** Mathematically align irregular discrete text events with regular 5-min price bars using: S(t) = sum S_i * exp(-lambda * dt).  
**Kabul Kriterleri:**
- [ ] `src/data_alignment/exponential_decay.py` with `TemporalAligner`
- [ ] Configurable lambda (default: 0.5 per hour)
- [ ] Vectorized numpy computation without slow Python loops

#### **Issue #3-7**
**Title:** `[Dataset] Multi-Modal PyTorch Aligned Dataset & Walk-Forward DataLoader`  
**Labels:** `deep-learning`, `data-engine`  
**Amaç & Kapsam:** Construct PyTorch Dataset yielding tuple pairs (price_seq [Batch, 78, 16], text_emb [Batch, 768], label [Batch]) with chronological walk-forward splits.  
**Kabul Kriterleri:**
- [ ] `src/data_alignment/dataset_builder.py`
- [ ] Sliding window construction (78 bars lookback)
- [ ] Chronological train/val/test splits

---

## 🧠 PHASE 4: ML Baselines & Hybrid Deep Learning

#### **Issue #4-1**
**Title:** `[ML] HistGradientBoosting & RandomForest Baseline Classifiers`  
**Labels:** `deep-learning`, `data-science`  
**Amaç & Kapsam:** Implement Scikit-Learn tabular baseline classifiers for benchmark comparison.  
**Kabul Kriterleri:**
- [ ] `src/models/baseline_models.py` with `BaselineModelTrainer`
- [ ] Compute PR-AUC, ROC-AUC, F1-score, and Brier score

#### **Issue #4-2**
**Title:** `[DL] PyTorch Bi-LSTM & Temporal Convolutional Network (TCN) Time-Series Branch`  
**Labels:** `deep-learning`, `backend`  
**Amaç & Kapsam:** Build the time-series encoder branch in PyTorch with Bi-LSTM and causal dilated TCN options.  
**Kabul Kriterleri:**
- [ ] `src/models/time_series_branch.py` with `TimeSeriesEncoder`
- [ ] LayerNorm, Dropout, and residual connections

#### **Issue #4-3**
**Title:** `[DL] PyTorch NLP FinBERT Dense Projection Branch`  
**Labels:** `deep-learning`, `nlp`  
**Amaç & Kapsam:** Map 768-D FinBERT embeddings into the common multi-modal latent dimension (128-D).  
**Kabul Kriterleri:**
- [ ] `src/models/text_branch.py` with `TextProjectionEncoder`
- [ ] Linear -> LayerNorm -> GELU -> Dropout

#### **Issue #4-4**
**Title:** `[DL] Multi-Head Cross-Attention Hybrid Fusion Layer (MarketPulseNet)`  
**Labels:** `deep-learning`, `backend`  
**Amaç & Kapsam:** Fuse text representation (Query) and price sequence features (Key/Value) using Multi-Head Cross-Attention.  
**Kabul Kriterleri:**
- [ ] `src/models/cross_attention.py` with `CrossAttentionFusion`
- [ ] `src/models/hybrid_network.py` with `MarketPulseNet`

#### **Issue #4-5**
**Title:** `[DL] Binary Focal Loss for Extreme Class Imbalance`  
**Labels:** `deep-learning`, `data-science`  
**Amaç & Kapsam:** Implement Binary Focal Loss to resolve 90-10 class imbalance in volatility spikes.  
**Kabul Kriterleri:**
- [ ] `src/models/loss_functions.py` with `BinaryFocalLoss`
- [ ] Parameters: gamma=2.0, alpha=0.75

#### **Issue #4-6**
**Title:** `[DL] PyTorch Model Training Engine with Checkpointing & Early Stopping`  
**Labels:** `deep-learning`, `backend`  
**Amaç & Kapsam:** Create training loop with AdamW, gradient clipping, validation PR-AUC monitoring, and model checkpointing.  
**Kabul Kriterleri:**
- [ ] `src/models/trainer.py` with `ModelTrainer`
- [ ] Checkpoint serialization to `models/checkpoints/best_marketpulse_net.pt`

#### **Issue #4-7**
**Title:** `[Eval] Model Evaluation & Hypothesis Testing Benchmark Suite`  
**Labels:** `deep-learning`, `data-science`  
**Amaç & Kapsam:** Benchmark hybrid model vs baselines using Diebold-Mariano and PR-AUC statistical evaluation.  
**Kabul Kriterleri:**
- [ ] `scripts/evaluate_benchmark.py`
- [ ] Detailed performance comparison tables

---

## 🔍 PHASE 5: Explainable AI (XAI) & Attribution

#### **Issue #5-1**
**Title:** `[XAI] SHAP Feature Importance Explainer for Baseline & Time-Series Features`  
**Labels:** `xai`, `data-science`  
**Amaç & Kapsam:** Compute SHAP values for tabular and time-series feature contributions to volatility forecasts.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/shap_explainer.py` with `ShapExplainer`
- [ ] Feature importance ranking output

#### **Issue #5-2**
**Title:** `[XAI] Captum Integrated Gradients & Cross-Attention Attribution`  
**Labels:** `xai`, `deep-learning`  
**Amaç & Kapsam:** Implement Integrated Gradients via Captum to attribute predictions to specific time intervals and NLP sentiment dimensions.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/integrated_gradients.py` with `IntegratedGradientsExplainer`
- [ ] Cross-attention heatmap weight extraction

#### **Issue #5-3**
**Title:** `[XAI] Volatility Risk Factor Decomposition Service`  
**Labels:** `xai`, `backend`  
**Amaç & Kapsam:** Decompose prediction risk into institutional percentages: e.g. 65% Breaking News, 20% RSI Divergence, 15% Volume Surge.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/attribution_service.py` with `RiskAttributionService`
- [ ] Human-readable narrative generation

---

## 🚀 PHASE 6: Production FastAPI Backend

#### **Issue #6-1**
**Title:** `[API] FastAPI Application Skeleton & Pydantic v2 Schemas`  
**Labels:** `fastapi`, `backend`  
**Amaç & Kapsam:** Build the production FastAPI application, CORS middleware, and Pydantic v2 request/response schemas.  
**Kabul Kriterleri:**
- [ ] `src/api/main.py` and `src/api/schemas.py`
- [ ] Schema validation for `PredictRequest`, `ExplainResponse`, etc.

#### **Issue #6-2**
**Title:** `[API] /predict Real-Time Multi-Modal Volatility Inference Endpoint`  
**Labels:** `fastapi`, `backend`  
**Amaç & Kapsam:** Serve sub-50ms volatility spike predictions combining incoming OHLCV bars and text headlines.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/predict.py`
- [ ] Output includes probability, risk level (Low/Moderate/Critical), confidence interval, and latency

#### **Issue #6-3**
**Title:** `[API] /explain Real-Time SHAP & Attribution Decomposition Endpoint`  
**Labels:** `fastapi`, `xai`  
**Amaç & Kapsam:** Serve on-demand XAI attribution and factor breakdown for any prediction ID.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/explain.py`

#### **Issue #6-4**
**Title:** `[API] /backtest Strategy Simulation Endpoint`  
**Labels:** `fastapi`, `backend`  
**Amaç & Kapsam:** Simulate risk-managed hedging strategy against Buy & Hold benchmark over historical bars.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/backtest.py` and `src/models/backtester.py`
- [ ] Computes Sharpe, Sortino, Max Drawdown, and equity series

#### **Issue #6-5**
**Title:** `[API] Prometheus Metrics, Rate Limiting & Health Check Middleware`  
**Labels:** `fastapi`, `infrastructure`  
**Amaç & Kapsam:** Expose `/health` and `/metrics` endpoints for Kubernetes probes and Prometheus metrics scraping.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/health.py`

---

## 🖥️ PHASE 7: Streamlit Financial Terminal Dashboard

#### **Issue #7-1**
**Title:** `[Dashboard] Modern Dark Glassmorphic Dashboard Layout & Navigation`  
**Labels:** `dashboard`, `frontend`  
**Amaç & Kapsam:** Develop an institutional dark glassmorphic terminal theme and sidebar navigation in Streamlit.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/app.py` with custom CSS theme

#### **Issue #7-2**
**Title:** `[Dashboard] Real-Time Volatility Monitor & Live Risk Gauge Component`  
**Labels:** `dashboard`, `frontend`  
**Amaç & Kapsam:** Build real-time volatility gauge and 5-minute interactive candlestick chart with Plotly.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/1_Realtime_Monitor.py`

#### **Issue #7-3**
**Title:** `[Dashboard] Explainability Explorer with SHAP Waterfall & Attention Heatmaps`  
**Labels:** `dashboard`, `xai`  
**Amaç & Kapsam:** Interactive visualizer showing factor breakdown pie chart, headline details, and SHAP waterfall.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/2_Explainability_Explorer.py`

#### **Issue #7-4**
**Title:** `[Dashboard] Backtesting & Strategy Performance Engine`  
**Labels:** `dashboard`, `frontend`  
**Amaç & Kapsam:** Interactive backtesting dashboard displaying equity curves, drawdown reductions, and quantitative metrics.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/3_Backtesting_Engine.py`

#### **Issue #7-5**
**Title:** `[Dashboard] System Health, Pipeline Freshness & Model Drift Monitor`  
**Labels:** `dashboard`, `infrastructure`  
**Amaç & Kapsam:** Dashboard page displaying pipeline synchronization status, API latency histogram, and model architecture specs.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/5_System_Health.py`

---

## 📦 PHASE 8: Documentation, Testing & Release

#### **Issue #8-1**
**Title:** `[Docs] Complete Technical Documentation Suite`  
**Labels:** `documentation`  
**Amaç & Kapsam:** Author full technical markdown guides covering architecture, data pipeline, model design, API reference, deployment, and troubleshooting.  
**Kabul Kriterleri:**
- [ ] `docs/ARCHITECTURE.md`, `docs/DATA_PIPELINE.md`, `docs/MODEL_DESIGN.md`, `docs/API_REFERENCE.md`, `docs/DEPLOYMENT_GUIDE.md`, `docs/TROUBLESHOOTING.md`

#### **Issue #8-2**
**Title:** `[Test] Comprehensive Unit & Integration Test Suite`  
**Labels:** `documentation`, `backend`  
**Amaç & Kapsam:** Build comprehensive unit and integration tests across configuration, feature engineering, alignment, models, and REST endpoints.  
**Kabul Kriterleri:**
- [ ] `tests/test_config.py`, `tests/test_feature_engineering.py`, `tests/test_alignment.py`, `tests/test_models.py`, `tests/test_api.py`

#### **Issue #8-3**
**Title:** `[DevOps] Kubernetes Manifests, Helm Charts & Production Deployment Guide`  
**Labels:** `devops`, `infrastructure`  
**Amaç & Kapsam:** Configure production Kubernetes Deployment and Service manifests for automated cluster deployments.  
**Kabul Kriterleri:**
- [ ] `deploy/kubernetes/deployment-api.yaml` and `deploy/kubernetes/service.yaml`

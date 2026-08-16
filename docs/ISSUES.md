# MarketPulse AI - Project Issues & Task Catalog 📋

This document catalogues all 28 project issues organized across 8 sprints for the **MarketPulse AI** repository.

---

## 🏗️ Sprint 1: Architecture, Infrastructure & DevOps

#### **Issue #1-1**
**Title:** `[Infra] Configuration Management & Structured Logging Engine`  
**Labels:** `infrastructure`, `backend`, `sprint-1`  
**Amaç & Kapsam:** Implement centralized environment configuration via Pydantic `BaseSettings` supporting YAML and `.env` hierarchies, along with structured JSON logging.  
**Kabul Kriterleri:**
- [ ] `src/config/settings.py` with `Settings` class loading `config/default.yaml` and `.env`
- [ ] `src/utils/logger.py` producing JSON structured log output with correlation IDs
- [ ] Unit tests for configuration overriding and invalid parameter handling

#### **Issue #1-2**
**Title:** `[DevOps] Multi-Stage Dockerfile & Local docker-compose Stack`  
**Labels:** `devops`, `docker`, `sprint-1`  
**Amaç & Kapsam:** Build an optimized multi-stage `Dockerfile` and `docker-compose.yml` to run the FastAPI backend, Streamlit dashboard, and Prometheus locally.  
**Kabul Kriterleri:**
- [ ] Multi-stage `Dockerfile` targeting minimal image size (< 1.5GB with PyTorch/Transformers)
- [ ] `docker-compose.yml` defining `api`, `dashboard`, and `prometheus` services with network isolation
- [ ] Health checks configured for all containers

#### **Issue #1-3**
**Title:** `[CI/CD] GitHub Actions Automated Testing, Linting & Security Gateways`  
**Labels:** `devops`, `ci-cd`, `sprint-1`  
**Amaç & Kapsam:** Establish GitHub Actions workflows to enforce Black, isort, Flake8, Mypy, Bandit security scanning, and pytest coverage across Python 3.10 and 3.11.  
**Kabul Kriterleri:**
- [ ] `.github/workflows/ci.yml` matrix pipeline
- [ ] Code formatting & lint check steps
- [ ] Unit test execution with coverage artifact publishing

---

## 📡 Sprint 2: Data Ingestion & Storage Lake

#### **Issue #2-1**
**Title:** `[Data] Alpaca Markets & Yahoo Finance 5-Minute OHLCV Data Collector`  
**Labels:** `data-engine`, `timeseries`, `sprint-2`  
**Amaç & Kapsam:** Build resilient connectors for historical and streaming 5-minute OHLCV bar data with rate limiting, retries, and automatic fallback to Yahoo Finance (`yfinance`).  
**Kabul Kriterleri:**
- [ ] `src/data_engine/alpaca_connector.py` with `AlpacaDataCollector`
- [ ] `src/data_engine/yfinance_connector.py` fallback adapter
- [ ] Pagination and rate-limit backoff handler
- [ ] Returns validated pandas DataFrame with standard columns: `[timestamp, open, high, low, close, volume, vwap, trade_count]`

#### **Issue #2-2**
**Title:** `[Data] Reddit PRAW Sentiment Scraper (r/wallstreetbets, r/stocks)`  
**Labels:** `data-engine`, `nlp`, `sprint-2`  
**Amaç & Kapsam:** Ingest financial discussions and posts from targeted subreddits using PRAW with ticker symbol extraction and timestamp normalization.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/reddit_collector.py` with `RedditCollector`
- [ ] Filters posts/comments mentioning target symbols (`SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `TSLA`)
- [ ] Normalizes UTC timestamps to ISO 8601

#### **Issue #2-3**
**Title:** `[Data] NewsAPI & GDELT Financial Headline Ingestion Service`  
**Labels:** `data-engine`, `nlp`, `sprint-2`  
**Amaç & Kapsam:** Collect streaming and historical financial news articles and regulatory alerts from NewsAPI and GDELT.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/news_collector.py` with `NewsCollector`
- [ ] Deduplication based on title similarity & URL hashes
- [ ] Graceful fallback for API quota limits

#### **Issue #2-4**
**Title:** `[Data] Parquet Storage Lake with Partitioning & Schema Validation`  
**Labels:** `data-engine`, `storage`, `sprint-2`  
**Amaç & Kapsam:** Implement local and cloud-ready Parquet data lake manager with date/ticker partitioning and pyarrow schema validation.  
**Kabul Kriterleri:**
- [ ] `src/data_engine/storage_manager.py` with `StorageManager`
- [ ] Partitions data: `data/raw/{source}/{symbol}/year={YYYY}/month={MM}/data.parquet`
- [ ] Read/write performance benchmarks (< 100ms for 100k rows)

---

## ⚙️ Sprint 3: Feature Engineering, Labeling & Temporal Alignment

#### **Issue #3-1**
**Title:** `[Features] Technical Indicators Engine (ATR, RSI, MACD, Bollinger, Rolling Volatility)`  
**Labels:** `feature-engineering`, `finance`, `sprint-3`  
**Amaç & Kapsam:** Compute a comprehensive suite of technical indicators without lookahead bias.  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/technical_indicators.py` with `TechnicalFeatureEngine`
- [ ] Implements ATR(14), RSI(14), MACD(12,26,9), Bollinger Bands(20,2), Rolling Volatility(12, 36, 78 bars), VWAP divergence
- [ ] Unit tests verifying calculation against reference values

#### **Issue #3-2**
**Title:** `[Labels] ATR Volatility Spike Ground-Truth Labeling Generator`  
**Labels:** `feature-engineering`, `modeling`, `sprint-3`  
**Amaç & Kapsam:** Generate binary target labels $y_t \in \{0, 1\}$ defined as $\max(\text{ATR}_{t+1:t+6}) \ge 1.15 \cdot \text{ATR}_t$.  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/labeler.py` with `VolatilityLabeler`
- [ ] Configurable forecast window (default: 6 bars / 30 mins) and threshold multiplier (default: 1.15)
- [ ] Correctly prevents lookahead leak during validation/test splits

#### **Issue #3-3**
**Title:** `[NLP] Financial Text Cleaning, Normalization & Tokenization Pipeline`  
**Labels:** `nlp`, `feature-engineering`, `sprint-3`  
**Amaç & Kapsam:** Clean and prepare financial texts (URLs, emojis, ticker cashtags `$AAPL`, contractions, HTML tags).  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/text_preprocessor.py` with `TextPreprocessor`
- [ ] URL extraction, cashtag preservation, whitespace normalization
- [ ] Length constraints (10 to 1000 characters)

#### **Issue #3-4**
**Title:** `[NLP] FinBERT Sentiment 768-D Embedding Generation & Cache Engine`  
**Labels:** `nlp`, `deep-learning`, `sprint-3`  
**Amaç & Kapsam:** Extract 768-dimensional sentence embeddings using pre-trained FinBERT (`ProsusAI/finbert`) with GPU batching and local caching.  
**Kabul Kriterleri:**
- [ ] `src/feature_engineering/sentiment_embedder.py` with `FinBERTEmbedder`
- [ ] Batch tokenization with max length 512, truncation, CLS token extraction
- [ ] Disk cache for precomputed embeddings

#### **Issue #3-5**
**Title:** `[Alignment] Exponential Decay Forward-Fill Temporal Alignment Engine`  
**Labels:** `feature-engineering`, `algorithms`, `sprint-3`  
**Amaç & Kapsam:** Mathematically fuse irregular discrete text events into continuous 5-minute price bars using exponential decay: $S(t) = \sum S_i \cdot e^{-\lambda \Delta t}$.  
**Kabul Kriterleri:**
- [ ] `src/data_alignment/exponential_decay.py` with `TemporalAligner`
- [ ] Vectorized decay calculation with half-life parameterization ($\lambda = 0.5/\text{hour}$)
- [ ] Unit tests verifying decay curve values

#### **Issue #3-6**
**Title:** `[Dataset] Multi-Modal PyTorch Aligned Dataset & Walk-Forward DataLoader`  
**Labels:** `deep-learning`, `data-alignment`, `sprint-3`  
**Amaç & Kapsam:** Build a PyTorch Dataset yielding tuple pairs `(price_sequence_tensor, text_embedding_tensor, target_label)` with walk-forward time-series split.  
**Kabul Kriterleri:**
- [ ] `src/data_alignment/dataset_builder.py` with `MultiModalDataset`
- [ ] Shape verification: Price `[Batch, 78, 16]`, Text `[Batch, 768]`, Label `[Batch]`
- [ ] Purged walk-forward splitting to prevent train-test contamination

---

## 🧠 Sprint 4: ML Baselines & Hybrid Deep Learning Architecture

#### **Issue #4-1**
**Title:** `[ML] HistGradientBoosting & LightGBM Baseline Models with Purged CV`  
**Labels:** `machine-learning`, `baseline`, `sprint-4`  
**Amaç & Kapsam:** Implement Scikit-Learn `HistGradientBoostingClassifier` and `RandomForestClassifier` baselines for comparison.  
**Kabul Kriterleri:**
- [ ] `src/models/baseline_models.py` with `BaselineModelTrainer`
- [ ] Purged K-Fold Cross Validation
- [ ] Computes baseline AUC-ROC, PR-AUC, F1-Score, Brier score

#### **Issue #4-2**
**Title:** `[DL] PyTorch Bi-LSTM & Temporal Convolutional Network (TCN) Time-Series Branch`  
**Labels:** `deep-learning`, `pytorch`, `sprint-4`  
**Amaç & Kapsam:** Build the time-series encoder branch in PyTorch with Bi-LSTM and causal dilated TCN options.  
**Kabul Kriterleri:**
- [ ] `src/models/time_series_branch.py` with `TimeSeriesEncoder` (BiLSTM + TCN)
- [ ] Layer normalization, dropout, residual connections

#### **Issue #4-3**
**Title:** `[DL] PyTorch NLP FinBERT Dense Projection Branch`  
**Labels:** `deep-learning`, `pytorch`, `sprint-4`  
**Amaç & Kapsam:** Construct the text feature projection module mapping 768-D FinBERT vectors to the common multi-modal latent space (128-D).  
**Kabul Kriterleri:**
- [ ] `src/models/text_branch.py` with `TextProjectionEncoder`
- [ ] Linear -> BatchNorm/LayerNorm -> GELU -> Dropout

#### **Issue #4-4**
**Title:** `[DL] Multi-Head Cross-Attention Hybrid Fusion Layer (MarketPulseNet)`  
**Labels:** `deep-learning`, `architecture`, `sprint-4`  
**Amaç & Kapsam:** Combine time-series and NLP representations via Multi-Head Cross-Attention where text queries price context and vice-versa.  
**Kabul Kriterleri:**
- [ ] `src/models/cross_attention.py` with `CrossAttentionFusion`
- [ ] `src/models/hybrid_network.py` with end-to-end `MarketPulseNet`
- [ ] Classification head with sigmoid output

#### **Issue #4-5**
**Title:** `[DL] PyTorch Model Training Engine with Focal Loss, LR Scheduler & Checkpointing`  
**Labels:** `deep-learning`, `training`, `sprint-4`  
**Amaç & Kapsam:** Comprehensive PyTorch training pipeline with Focal Loss (for class imbalance), AdamW, CosineAnnealingLR, gradient clipping, and early stopping.  
**Kabul Kriterleri:**
- [ ] `src/models/loss_functions.py` with `FocalLoss`
- [ ] `src/models/trainer.py` with `ModelTrainer`
- [ ] Checkpoint saving: best model based on validation PR-AUC

#### **Issue #4-6**
**Title:** `[Eval] Model Evaluation & Hypothesis Testing Benchmark Suite`  
**Labels:** `evaluation`, `benchmarking`, `sprint-4`  
**Amaç & Kapsam:** Benchmark hybrid model vs ML baselines using statistical tests (Diebold-Mariano test, McNemar test).  
**Kabul Kriterleri:**
- [ ] Comprehensive performance tables: AUC-ROC, PR-AUC, F1, Precision@K
- [ ] Hypothesis testing scripts

---

## 🔍 Sprint 5: Explainable AI (XAI) & Attribution Layer

#### **Issue #5-1**
**Title:** `[XAI] SHAP Explainer for Baseline & Time-Series Features`  
**Labels:** `xai`, `interpretability`, `sprint-5`  
**Amaç & Kapsam:** Compute SHAP values for tabular and time-series features to generate waterfall and summary plots.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/shap_explainer.py` with `ShapExplainer`
- [ ] Generates feature importance rankings

#### **Issue #5-2**
**Title:** `[XAI] Captum Integrated Gradients & Attention Attribution for Hybrid Network`  
**Labels:** `xai`, `captum`, `sprint-5`  
**Amaç & Kapsam:** Implement Integrated Gradients with Captum to attribute predictions to individual time steps and NLP sentiment dimensions.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/integrated_gradients.py` with `IntegratedGradientsExplainer`
- [ ] Extracts cross-attention weights for visualization

#### **Issue #5-3**
**Title:** `[XAI] Volatility Risk Factor Decomposition Service (News vs Technical Impact)`  
**Labels:** `xai`, `backend`, `sprint-5`  
**Amaç & Kapsam:** Decompose volatility risk score into percentage contributions: e.g., 65% Breaking News, 20% RSI Divergence, 15% Volume Surge.  
**Kabul Kriterleri:**
- [ ] `src/xai_explainer/attribution_service.py` with `RiskAttributionService`
- [ ] Standardized JSON output for API & UI consumption

---

## 🚀 Sprint 6: Production FastAPI Backend & Microservices

#### **Issue #6-1**
**Title:** `[API] FastAPI Application Skeleton & Pydantic Schemas`  
**Labels:** `api`, `fastapi`, `sprint-6`  
**Amaç & Kapsam:** Build the production FastAPI app structure, Pydantic v2 schemas, exception handlers, and CORS/logging middleware.  
**Kabul Kriterleri:**
- [ ] `src/api/main.py` with FastAPI instance & lifespan handlers
- [ ] `src/api/schemas.py` with `PredictRequest`, `PredictResponse`, `ExplainResponse`, `BacktestResponse`

#### **Issue #6-2**
**Title:** `[API] /predict, /explain & /backtest Real-Time Inference Endpoints`  
**Labels:** `api`, `fastapi`, `sprint-6`  
**Amaç & Kapsam:** Implement high-performance prediction and explanation REST endpoints with batching and caching.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/predict.py` (< 50ms latency target)
- [ ] `src/api/routes/explain.py`
- [ ] `src/api/routes/backtest.py`

#### **Issue #6-3**
**Title:** `[API] Prometheus Metrics, Rate Limiting & Health Check Middleware`  
**Labels:** `api`, `monitoring`, `sprint-6`  
**Amaç & Kapsam:** Expose `/metrics` for Prometheus, implement `/health` readiness/liveness checks, and API rate limiting.  
**Kabul Kriterleri:**
- [ ] `src/api/routes/health.py`
- [ ] Prometheus metrics: request latency, prediction distribution, model inference counter

---

## 🖥️ Sprint 7: Interactive Financial Terminal Dashboard (Streamlit)

#### **Issue #7-1**
**Title:** `[Dashboard] Modern Dark Glassmorphic Dashboard Layout & Navigation`  
**Labels:** `dashboard`, `frontend`, `sprint-7`  
**Amaç & Kapsam:** Create an institutional-grade dark theme Streamlit interface with sidebar navigation and reactive state.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/app.py` with multi-page navigation
- [ ] Custom CSS styling (Bloomberg/TradingView terminal aesthetics)

#### **Issue #7-2**
**Title:** `[Dashboard] Real-Time Volatility Monitor & Live Risk Gauge`  
**Labels:** `dashboard`, `visualization`, `sprint-7`  
**Amaç & Kapsam:** Build interactive live volatility gauge (Low / Moderate / Critical Volatility Risk) and candlestick charts with Plotly.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/1_Realtime_Monitor.py`
- [ ] Live risk gauge with confidence interval bands

#### **Issue #7-3**
**Title:** `[Dashboard] Explainability Explorer (SHAP Waterfall & Factor Attribution)`  
**Labels:** `dashboard`, `xai`, `sprint-7`  
**Amaç & Kapsam:** Interactive XAI panel showing SHAP waterfall charts, attention heatmaps, and news headline impact bars.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/2_Explainability_Explorer.py`
- [ ] Factor percentage breakdown cards

#### **Issue #7-4**
**Title:** `[Dashboard] Backtesting & Strategy Performance Engine (Equity Curve & Drawdowns)`  
**Labels:** `dashboard`, `finance`, `sprint-7`  
**Amaç & Kapsam:** Visual backtesting tool comparing volatility-avoidance trading strategies against buy-and-hold benchmarks.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/3_Backtesting_Engine.py`
- [ ] Cumulative returns, max drawdown, Sharpe, Sortino, Calmar ratio displays

#### **Issue #7-5**
**Title:** `[Dashboard] System Health, Pipeline Freshness & Model Drift Monitor`  
**Labels:** `dashboard`, `monitoring`, `sprint-7`  
**Amaç & Kapsam:** Dashboard page tracking data feed freshness, API response times, memory consumption, and prediction drift.  
**Kabul Kriterleri:**
- [ ] `src/dashboard/pages/5_System_Health.py`
- [ ] Real-time status indicators for Alpaca, Reddit, and News pipelines

---

## 📦 Sprint 8: Documentation, Verification & Git Integration

#### **Issue #8-1**
**Title:** `[Docs] Complete Technical Documentation Suite (Architecture, Data, Models, API)`  
**Labels:** `documentation`, `sprint-8`  
**Amaç & Kapsam:** Complete all technical markdown guides in `docs/`.  
**Kabul Kriterleri:**
- [ ] `ARCHITECTURE.md`, `DATA_PIPELINE.md`, `MODEL_DESIGN.md`, `API_REFERENCE.md`, `DEPLOYMENT_GUIDE.md`, `TROUBLESHOOTING.md`

#### **Issue #8-2**
**Title:** `[Test] Comprehensive Unit & Integration Test Suite (>80% coverage)`  
**Labels:** `testing`, `quality`, `sprint-8`  
**Amaç & Kapsam:** Author comprehensive unit and integration tests across data ingestion, feature engineering, modeling, and API endpoints.  
**Kabul Kriterleri:**
- [ ] Pytest suite passing with `pytest tests/`

#### **Issue #8-3**
**Title:** `[Git] Push project skeleton, documentation, roadmap, and issue templates to GitHub`  
**Labels:** `git`, `release`, `sprint-8`  
**Amaç & Kapsam:** Commit and push all structured codebase files, documentation, and configuration to GitHub `main` branch.  
**Kabul Kriterleri:**
- [ ] Clean git commit history
- [ ] Remote repository updated

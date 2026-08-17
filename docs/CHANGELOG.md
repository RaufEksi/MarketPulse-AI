# Changelog

All notable changes to MarketPulse AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.4.0] - 2026-08-17

### Added — Sprint 4: ML Baselines & Hybrid Deep Learning Architecture
- **Model Training Engine** (`src/models/trainer.py`): PyTorch training loop with Binary Focal Loss, AdamW optimizer, CosineAnnealingLR scheduler, early stopping patience, and validation PR-AUC checkpointing
- **Time-Series Branch** (`src/models/time_series_branch.py`): Bidirectional LSTM and Temporal Convolutional Network (TCN) encoders with causal dilated residual blocks
- **Cross-Attention Fusion** (`src/models/cross_attention.py` + `src/models/hybrid_network.py`): MarketPulseNet end-to-end multi-modal architecture with attention maps
- **ML Baseline Suite** (`src/models/baseline_models.py`): HistGradientBoosting and RandomForest tabular benchmarks
- **Volatility Backtester** (`src/models/backtester.py`): Quantitative strategy simulator for volatility avoidance and risk metrics calculation
- **Standalone Execution Scripts**: `scripts/train_model.py`, `scripts/evaluate_benchmark.py`, `scripts/run_backtest.py`

---

## [0.3.0] - 2026-08-17

### Added — Sprint 3: Feature Engineering & Temporal Alignment
- **VWAP Divergence & Micro-Structure** (`src/feature_engineering/technical_indicators.py`): VWAP deviation ratio and volume ratio calculation
- **Technical Indicators Engine**: 14-period ATR, RSI, MACD line/signal/hist, Bollinger Bands (%B & Bandwidth), Rolling Volatility (12, 36, 78 bars)
- **Volatility Spike Ground-Truth Labeler** (`src/feature_engineering/labeler.py`): Binary target generation ($\max(\text{ATR}_{t+1:t+6}) \ge 1.15 \cdot \text{ATR}_t$) without lookahead bias
- **Text Cleaning & FinBERT Embeddings** (`src/feature_engineering/text_preprocessor.py`, `sentiment_embedder.py`): 768-D CLS token extraction with lazy loading and fallback support
- **Vectorized Exponential Decay Alignment** (`src/data_alignment/exponential_decay.py`): Mathematical forward-fill for asynchronous sentiment signals
- **Walk-Forward DataLoader** (`src/data_alignment/dataset_builder.py`): Chronological sequence window builder and PyTorch DataLoader factory
- **Comprehensive Automated Test Suite**: 27 unit & integration tests across all modules achieving **86% code coverage**

---

## [0.2.0] - 2026-08-15

### Added — Sprint 2: Data Ingestion & Storage Lake
- **Alpaca Markets connector** (`src/data_engine/alpaca_connector.py`): AlpacaDataCollector with rate-limit backoff and pagination for 5-minute OHLCV bars
- **Yahoo Finance fallback** (`src/data_engine/yfinance_connector.py`): YFinanceDataCollector with automatic column schema harmonization
- **Reddit PRAW scraper** (`src/data_engine/reddit_collector.py`): RedditCollector filtering `r/wallstreetbets`, `r/stocks`, `r/investing` by target symbols
- **NewsAPI headline collector** (`src/data_engine/news_collector.py`): NewsCollector with headline deduplication by URL and content hash
- **Parquet storage manager** (`src/data_engine/storage_manager.py`): StorageManager with symbol/source/date partitioning
- Data directory structure: `data/raw/`, `data/processed/`, `data/cache/`

---

## [0.1.0] - 2026-08-07

### Added — Sprint 1: Infrastructure, Config & DevOps Engine
- **Project scaffolding**: Complete repository structure with `src/`, `tests/`, `docs/`, `scripts/`, `config/`, `deploy/`
- **Configuration management** (`src/config/settings.py`): Pydantic BaseSettings with YAML loader, environment-specific overrides, and `@lru_cache` singleton
- **YAML configs**: `config/default.yaml`, `config/development.yaml`, `config/production.yaml`
- **Structured logging** (`src/utils/logger.py`): JSONFormatter with ISO8601 timestamps and execution metadata
- **Exception hierarchy** (`src/utils/exceptions.py`): MarketPulseException base with DataIngestionError, ModelInferenceError
- **Financial metrics** (`src/utils/metrics.py`): PR-AUC, ROC-AUC, F1, Brier score, Sharpe/Sortino/Max Drawdown utilities
- **Multi-stage Dockerfile** and `docker-compose.yml` with API + Dashboard services
- **Makefile**: setup, install, test, lint, format, type-check, serve, docker tasks
- **GitHub Actions CI/CD** (`.github/workflows/ci.yml`): Matrix testing (Python 3.10, 3.11) with Black, isort, Flake8, Bandit, pytest
- **Issue templates**: bug_report.md, epic_task.md, pull_request_template.md
- **pyproject.toml**: Full tool configuration (black, isort, mypy, pytest, coverage)
- **Documentation**: README.md, CONTRIBUTING.md, ARCHITECTURE.md, MODEL_DESIGN.md, DATA_PIPELINE.md, API_REFERENCE.md, DEPLOYMENT_GUIDE.md, TROUBLESHOOTING.md, ROADMAP.md, ISSUES.md

### Added — Sprint 3-7: Scaffolded Modules (not yet integration-tested)
- **Feature Engineering**: TechnicalFeatureEngine, VolatilityLabeler, TextPreprocessor, FinBERTEmbedder
- **Temporal Alignment**: TemporalAligner (exponential decay), AlignedDataset (PyTorch Dataset)
- **Models**: TimeSeriesEncoder (Bi-LSTM/TCN), TextProjectionEncoder, CrossAttentionFusion, MarketPulseNet, BinaryFocalLoss, ModelTrainer, Backtester, BaselineModelTrainer
- **XAI**: ShapExplainer, IntegratedGradientsExplainer, RiskAttributionService
- **API**: FastAPI application skeleton with Pydantic v2 schemas
- **Dashboard**: Streamlit application skeleton with page structure

---

## Version Reference

- `0.1.0` — Sprint 1 completion (Infrastructure & DevOps)
- `0.2.0` — Sprint 2 completion (Data Ingestion & Storage)
- `0.3.0` — Sprint 3 target (Feature Engineering & Alignment)
- `0.4.0` — Sprint 4 target (ML Baselines & Hybrid DL)
- `0.5.0` — Sprint 5 target (XAI & Attribution)
- `0.6.0` — Sprint 6 target (FastAPI Backend)
- `0.7.0` — Sprint 7 target (Streamlit Dashboard)
- `1.0.0` — Sprint 8 (Testing, MLOps & Release)

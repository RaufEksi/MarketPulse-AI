# MarketPulse AI — Architecture Decision Records (ADR) 📝

This document records key technical decisions made during the development of MarketPulse AI, their context, alternatives considered, and rationale.

---

## ADR Index

| ID | Decision | Date | Status |
|----|----------|------|--------|
| ADR-001 | Bi-LSTM/TCN over Transformer for time series | 2026-08-01 | ✅ Accepted |
| ADR-002 | Focal Loss over Weighted BCE | 2026-08-01 | ✅ Accepted |
| ADR-003 | Exponential Decay over Simple Forward-Fill | 2026-08-01 | ✅ Accepted |
| ADR-004 | FinBERT over all-MiniLM for financial sentiment | 2026-08-01 | ✅ Accepted |
| ADR-005 | Parquet over SQLite/PostgreSQL for data storage | 2026-08-01 | ✅ Accepted |
| ADR-006 | Python 3.10+ minimum version | 2026-08-01 | ✅ Accepted |
| ADR-007 | Streamlit over Gradio/Dash for dashboard | 2026-08-01 | ✅ Accepted |
| ADR-008 | Multi-Head Cross-Attention for fusion | 2026-08-01 | ✅ Accepted |
| ADR-009 | ATR-based volatility spike definition | 2026-08-01 | ✅ Accepted |

---

## ADR-001: Bi-LSTM/TCN over Transformer for Time Series Encoding

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
The time series branch needs to encode 78-bar (one trading session) sequences of 16 technical features. We needed an architecture that captures temporal dependencies in 5-minute OHLCV data.

### Options Considered
1. **Transformer Encoder** — Self-attention over full sequence
2. **Bi-LSTM** — Bidirectional recurrent encoding
3. **Temporal Convolutional Network (TCN)** — Causal dilated convolutions
4. **Bi-LSTM + TCN (selected)** — Support for both, configurable

### Decision
Provide both **Bi-LSTM** and **TCN** as selectable encoder types via `config/default.yaml` (`model.time_series.model_type: "bilstm"` or `"tcn"`).

### Rationale
- **Sequence length is moderate** (78 bars) — Transformer self-attention overhead not justified
- **LSTM excels at ordered temporal dependencies** — financial time series are inherently sequential
- **TCN offers faster training** with causal convolutions — good for ablation comparison
- **Transformer would add model complexity** without proven benefit for 78-length sequences
- Supporting both enables **benchmarking** for the research paper

### Consequences
- Two encoder implementations to maintain (`time_series_branch.py`)
- Model selection via config, not code changes
- If sequence length increases significantly (>500 bars), revisit Transformer option

---

## ADR-002: Focal Loss over Weighted BCE for Class Imbalance

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
Volatility spikes occur in approximately 5-10% of observations, creating severe class imbalance (90-10 ratio). Standard BCE loss causes the model to predict all negatives.

### Options Considered
1. **Weighted BCE** — Manually set class weights
2. **Binary Focal Loss (selected)** — Dynamic downweighting of easy negatives
3. **SMOTE/oversampling** — Synthetic minority oversampling
4. **Asymmetric Loss** — Different penalties for FP vs FN

### Decision
Use **Binary Focal Loss** with γ=2.0, α=0.75.

### Rationale
- Focal Loss **automatically downweights** easy-to-classify normal market instances
- α=0.75 **penalizes false negatives more heavily** — missing a volatility spike is costlier than a false alarm
- Weighted BCE requires manual tuning of class weights per dataset
- SMOTE is inappropriate for time series data (violates temporal ordering)
- Focal Loss is standard in imbalanced classification (Lin et al., 2017)

### Consequences
- Custom loss function implementation in `src/models/loss_functions.py`
- γ and α are configurable via `config/default.yaml`
- Evaluation metric must be **PR-AUC** (not accuracy or ROC-AUC)

---

## ADR-003: Exponential Decay over Simple Forward-Fill for Temporal Alignment

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
News articles and Reddit posts arrive at irregular timestamps, while price data is sampled at regular 5-minute intervals. These must be aligned for multi-modal fusion.

### Options Considered
1. **Simple Forward-Fill** — Carry latest text embedding until next event
2. **Exponential Decay Forward-Fill (selected)** — Weight decays over time: S(t) = Σ Si · e^(-λΔt)
3. **Nearest-Neighbor Join** — Match each bar to closest text event
4. **Binning** — Aggregate all texts per 5-min window

### Decision
Use **Exponential Decay Forward-Fill** with λ=0.5 per hour (half-life ≈ 1.386 hours).

### Rationale
- News impact **decays over time** — a headline from 3 hours ago matters less than one from 5 minutes ago
- Simple forward-fill treats 5-minute-old and 5-hour-old news equally (unrealistic)
- Exponential decay is **mathematically principled** and used in financial event study literature
- Vectorized NumPy implementation achieves O(N) performance
- λ is configurable for experimentation

### Consequences
- `src/data_alignment/exponential_decay.py` implements vectorized decay
- λ tuning may be needed per data source (news vs Reddit)
- Requires accurate UTC timestamps on all text events

---

## ADR-004: FinBERT over all-MiniLM for Financial Sentiment

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
Text data from Reddit and news headlines needs to be converted to dense vector embeddings for the NLP branch of MarketPulseNet.

### Options Considered
1. **all-MiniLM-L6-v2** — General-purpose sentence transformer (384-D)
2. **FinBERT (selected)** — Financial domain pre-trained BERT (768-D)
3. **RoBERTa-base** — General-purpose language model (768-D)
4. **TF-IDF + SVD** — Traditional bag-of-words approach

### Decision
Use **FinBERT** (`ProsusAI/finbert`) with 768-D CLS token extraction.

### Rationale
- FinBERT is **pre-trained on financial text** (financial news, SEC filings, analyst reports)
- Understands financial jargon: "hawkish", "dovish", "liquidity crunch", "short squeeze"
- 768-D provides richer representation than 384-D all-MiniLM
- TF-IDF loses semantic meaning — "Fed raises rates" ≠ "Fed lowers rates" in TF-IDF
- FinBERT demonstrates **CV/portfolio differentiation** as a specialized domain model

### Consequences
- ~500MB model download on first run
- Requires GPU for efficient batch processing (CPU fallback available)
- Embedding cache in Parquet format (`data/cache/finbert_embeddings.parquet`)
- Higher latency than all-MiniLM (~3x slower per batch)

---

## ADR-005: Parquet over SQLite/PostgreSQL for Data Storage

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
Raw and processed data needs efficient storage with fast read performance for ML training pipelines.

### Options Considered
1. **SQLite** — Lightweight relational database
2. **PostgreSQL** — Full-featured RDBMS
3. **Parquet (selected)** — Columnar storage format
4. **HDF5** — Hierarchical data format

### Decision
Use **Apache Parquet** with PyArrow for all data storage, partitioned by source/symbol/date.

### Rationale
- **Columnar format** enables fast column-selective reads (only load needed features)
- **Native Pandas/PyArrow integration** — zero-copy reads into DataFrames
- **Partitioning** by symbol and date enables selective data loading
- **Compression** (Snappy/Gzip) reduces storage footprint
- No database server to manage — **zero infrastructure** overhead
- Sub-100ms load times for 100k rows
- PostgreSQL is overkill for a research/portfolio project without concurrent writes

### Consequences
- No SQL query capability — all filtering done in Pandas
- No ACID transactions — acceptable for append-only ML data
- Partition schema must be maintained manually

---

## ADR-006: Python 3.10+ Minimum Version

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
Select the minimum Python version for development and CI/CD.

### Options Considered
1. **Python 3.8** — Widest compatibility
2. **Python 3.10 (selected)** — Match operator, better type hints
3. **Python 3.12** — Latest stable with performance improvements

### Decision
Minimum **Python 3.10**, CI matrix tests on 3.10 and 3.11.

### Rationale
- Python 3.10 introduces `match` statement and improved `Union` type syntax (`X | Y`)
- PyTorch 2.0+ and HuggingFace Transformers 4.30+ fully support 3.10/3.11
- Python 3.12/3.13/3.14 have known **compatibility issues** with numpy, torch, and other scientific packages
- CI matrix ensures code works on both 3.10 and 3.11

### Consequences
- Developers must use pyenv or conda to install Python 3.10/3.11 if system Python is newer
- `requirements.txt` uses pinned versions tested against 3.10/3.11
- `pyproject.toml` specifies `requires-python = ">=3.10"`

---

## ADR-007: Streamlit over Gradio/Dash for Dashboard

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
An interactive financial terminal dashboard is needed to visualize real-time predictions, SHAP explanations, and backtesting results.

### Options Considered
1. **Gradio** — Quick ML demo interface
2. **Plotly Dash** — React-based analytical dashboards
3. **Streamlit (selected)** — Python-native data app framework
4. **React + FastAPI** — Full custom frontend

### Decision
Use **Streamlit** with Plotly for charting and custom CSS for glassmorphic dark theme.

### Rationale
- **Fastest iteration** — pure Python, no JavaScript required
- **Multi-page apps** natively supported
- **Plotly integration** for professional financial charts (candlestick, SHAP waterfall)
- Gradio is too limited for multi-page, complex dashboards
- Dash requires more boilerplate and React knowledge
- Custom React would take 3-4x longer to build
- Streamlit provides **institutional-grade** visual quality with custom CSS

### Consequences
- Custom CSS required for dark glassmorphic theme (not Streamlit default)
- Dashboard communicates with model via FastAPI REST endpoints (not direct import)
- Auto-refresh may need explicit `st.rerun()` or polling

---

## ADR-008: Multi-Head Cross-Attention for Modal Fusion

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
The time series encoder output [B, 78, 128] and text encoder output [B, 1, 128] need to be fused into a single representation for classification.

### Options Considered
1. **Concatenation + Dense** — Simple but ignores modal interaction
2. **Multi-Head Cross-Attention (selected)** — Text queries attend to price sequence
3. **Gated Bilinear Fusion** — Multiplicative interaction
4. **FiLM (Feature-wise Linear Modulation)** — Affine conditioning

### Decision
Use **Multi-Head Cross-Attention** where Text = Query, Price = Key/Value.

### Rationale
- Cross-attention enables **contextual interaction**: "Which price bars are most relevant to this news headline?"
- Attention weights are **interpretable** — directly visualizable as heatmaps for XAI
- Concatenation loses temporal context of price-text interaction
- 4-head attention with d_model=128 provides good capacity without overfitting
- Aligns with the Explainable AI goal — attention weights contribute to risk attribution

### Consequences
- Attention weight extraction needed for XAI visualization
- Computational cost slightly higher than concatenation
- Interpretability benefit justifies complexity

---

## ADR-009: ATR-Based Volatility Spike Definition

**Date**: 2026-08-01  
**Status**: ✅ Accepted

### Context
Define the prediction target (what is a "volatility spike"?).

### Options Considered
1. **Return-based threshold** — |return| > 2σ
2. **ATR expansion (selected)** — max(ATR[t+1:t+6]) ≥ 1.15 × ATR[t]
3. **VIX-based** — VIX change > threshold
4. **Realized volatility jump** — σ_realized > σ_expected

### Decision
A **volatility spike** occurs at bar t if: max(ATR[t+1:t+6]) ≥ 1.15 × ATR[t] (15% ATR increase over next 30 minutes / 6 bars).

### Rationale
- ATR is the **industry-standard** volatility measure for intraday trading
- 15% threshold captures **meaningful** regime shifts (not noise)
- Forward-looking 6-bar (30-min) window is actionable for hedging decisions
- Return-based thresholds are sensitive to price level — ATR normalizes automatically
- VIX is only available for SPX/SPY, not individual stocks
- Binary classification (spike/no-spike) is cleaner than regression

### Consequences
- Label generation must avoid **lookahead bias** — ATR[t+1:t+6] is ground truth, not feature
- Class imbalance (~5-10% positive) — addressed by Focal Loss (ADR-002)
- 15% threshold and 6-bar horizon are configurable in `config/default.yaml`

---

## Template for New ADRs

```markdown
## ADR-NNN: [Decision Title]

**Date**: YYYY-MM-DD  
**Status**: ✅ Accepted / 🔄 Proposed / ❌ Deprecated / 🔀 Superseded by ADR-XXX

### Context
[What is the problem or decision point?]

### Options Considered
1. [Option A]
2. [Option B (selected)]
3. [Option C]

### Decision
[What was decided?]

### Rationale
[Why was this option chosen over alternatives?]

### Consequences
[What are the implications, trade-offs, and follow-up actions?]
```

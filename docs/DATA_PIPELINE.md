# MarketPulse AI - Data Pipeline & Feature Engineering Guide 📊

This document details the data ingestion, preprocessing, temporal alignment formulas, and dataset construction mechanisms in MarketPulse AI.

---

## 1. Data Ingestion Architecture

### Time-Series Ingestion
- **Source**: Alpaca Markets Market Data API v2 (Fallback: `yfinance`).
- **Granularity**: 5-Minute Intraday OHLCV Bars.
- **Tickers**: `SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `TSLA`.
- **Trading Hours**: Regular US Market Hours (09:30 - 16:00 EST / 78 bars per session).

### NLP Sentiment Ingestion
- **Reddit Ingestion**: PRAW querying `r/wallstreetbets` and `r/stocks` for ticker-tagged submissions and high-engagement comments.
- **News Ingestion**: NewsAPI & GDELT polling financial news feeds, SEC alerts, and press releases.

---

## 2. Feature Engineering

### Technical Indicators (`TechnicalFeatureEngine`)
1. **Average True Range (ATR)**: $N=14$ periods.
2. **Relative Strength Index (RSI)**: $N=14$ periods.
3. **MACD**: Fast=12, Slow=26, Signal=9 (Difference & Histogram).
4. **Bollinger Bands**: Window=20, StdDev=2.0 (%B and Bandwidth).
5. **Rolling Realized Volatility**: Standard deviation of log returns over 12, 36, and 78 bars.
6. **Volume Weighted Average Price (VWAP)**: Ratio of close price to session VWAP.

### Volatility Spike Target Definition (`VolatilityLabeler`)
A binary volatility spike $y_t \in \{0, 1\}$ occurs at bar $t$ if:
$$\max_{k \in [1, 6]} \text{ATR}_{t+k} \ge 1.15 \cdot \text{ATR}_t$$
This corresponds to a $\ge 15\%$ expansion in market volatility within the subsequent 30 minutes (6 five-minute bars).

---

## 3. Mathematical Temporal Alignment Engine

Because news articles and social media comments arrive at irregular timestamps ($t_i$), they cannot be directly concatenated with regular 5-minute price bars.

MarketPulse AI applies an **Exponential Decay Forward-Fill** algorithm:

$$S(t_{\text{bar}}) = \sum_{i: t_i \le t_{\text{bar}}} S_i \cdot \exp\left(-\lambda \cdot (t_{\text{bar}} - t_i)\right)$$

Where:
- $S_i \in \mathbb{R}^{768}$: FinBERT sentence embedding of event $i$.
- $\Delta t = t_{\text{bar}} - t_i$: Elapsed time between event publication and bar timestamp.
- $\lambda$: Decay rate parameter (default: $\lambda = 0.5\text{ hour}^{-1}$, half-life $\approx 1.386$ hours).

---

## 4. Parquet Data Lake Structure

```
data/
├── raw/
│   ├── ohlcv/
│   │   └── symbol=SPY/year=2026/month=08/bars.parquet
│   └── text/
│       └── source=reddit/symbol=SPY/year=2026/month=08/posts.parquet
├── processed/
│   └── aligned_features_SPY_2026.parquet
└── cache/
    └── finbert_embeddings.parquet
```

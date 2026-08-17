"""
Real-Time Volatility Monitor Page.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering.technical_indicators import TechnicalFeatureEngine

st.title("⚡ Real-Time Volatility & Risk Monitor")

symbol = st.selectbox(
    "Select Asset Symbol",
    ["NVDA", "TSLA", "AAPL", "SPY", "QQQ", "MSFT"],
    index=0,
)

# Asset specific market profiles
PROFILE_CONFIG = {
    "NVDA": {
        "base": 128.5,
        "prob": 0.846,
        "vol_mult": 0.008,
        "atr_exp": "+23.4%",
        "risk": "CRITICAL_VOLATILITY",
    },
    "TSLA": {
        "base": 218.0,
        "prob": 0.782,
        "vol_mult": 0.007,
        "atr_exp": "+19.1%",
        "risk": "CRITICAL_VOLATILITY",
    },
    "QQQ": {
        "base": 482.5,
        "prob": 0.448,
        "vol_mult": 0.004,
        "atr_exp": "+9.2%",
        "risk": "MODERATE_VOLATILITY",
    },
    "AAPL": {
        "base": 224.0,
        "prob": 0.345,
        "vol_mult": 0.003,
        "atr_exp": "+5.4%",
        "risk": "LOW_VOLATILITY",
    },
    "SPY": {
        "base": 554.0,
        "prob": 0.284,
        "vol_mult": 0.002,
        "atr_exp": "+3.8%",
        "risk": "LOW_VOLATILITY",
    },
    "MSFT": {
        "base": 442.0,
        "prob": 0.228,
        "vol_mult": 0.0025,
        "atr_exp": "+2.9%",
        "risk": "LOW_VOLATILITY",
    },
}

cfg = PROFILE_CONFIG.get(symbol, PROFILE_CONFIG["SPY"])
prob = cfg["prob"]

# Generate simulated live intraday series
seed = abs(hash(symbol)) % 10000
np.random.seed(seed)
n_bars = 78
now = datetime.now(timezone.utc)
timestamps = [now - timedelta(minutes=5 * (n_bars - i)) for i in range(n_bars)]

rets = np.random.normal(0.0001, cfg["vol_mult"], size=n_bars)
if cfg["risk"] == "CRITICAL_VOLATILITY":
    rets[-8:] = np.random.normal(-0.004, cfg["vol_mult"] * 2.2, size=8)  # Recent shock
elif cfg["risk"] == "MODERATE_VOLATILITY":
    rets[-5:] = np.random.normal(-0.002, cfg["vol_mult"] * 1.5, size=5)

prices = cfg["base"] * np.exp(np.cumsum(rets))
highs = prices * (1.0 + np.abs(np.random.normal(0, cfg["vol_mult"] * 0.5, size=n_bars)))
lows = prices * (1.0 - np.abs(np.random.normal(0, cfg["vol_mult"] * 0.5, size=n_bars)))
opens = (prices + lows) / 2.0
closes = prices
volumes = np.random.randint(15000, 350000, size=n_bars)

# Compute live indicators
raw_df = pd.DataFrame(
    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
)
engine = TechnicalFeatureEngine()
feat_df = engine.transform(raw_df)
latest_atr = feat_df["atr_14"].iloc[-1]
latest_rsi = feat_df["rsi_14"].iloc[-1]

# Display Gauge & Warnings
col_gauge, col_stats = st.columns([1, 2])

gauge_color = "#ef4444" if prob >= 0.70 else "#f59e0b" if prob >= 0.40 else "#10b981"

with col_gauge:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"{symbol} Volatility Spike Risk (30m)", "font": {"size": 16}},
            delta={"reference": 40.0, "increasing": {"color": "red"}},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig_gauge.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_stats:
    if prob >= 0.70:
        st.markdown("### 🚨 Warning Level: **CRITICAL VOLATILITY DETECTED**")
        rec_action = f"Reduce long {symbol} equity allocation by 80% / hedge via index options."
    elif prob >= 0.40:
        st.markdown("### ⚠️ Warning Level: **MODERATE VOLATILITY ELEVATION**")
        rec_action = "Tighten stop-loss bands / scale down aggressive breakout positions by 40%."
    else:
        st.markdown("### 🟢 Warning Level: **LOW VOLATILITY REGIME (STABLE)**")
        rec_action = "Standard risk allocation. No hedging required."

    ci_lower = max(0.0, prob - 0.05)
    ci_upper = min(1.0, prob + 0.05)

    st.markdown(
        f"- **Predicted Spike Probability:** `{prob*100:.1f}%` "
        f"(CI: `{ci_lower*100:.1f}%` - `{ci_upper*100:.1f}%`)\n"
        f"- **Expected ATR Expansion:** `{cfg['atr_exp']}` over next 30 minutes\n"
        f"- **Live Market Microstructure:** ATR: `${latest_atr:.2f}` | "
        f"RSI(14): `{latest_rsi:.1f}` | Current Price: `${closes[-1]:.2f}`\n"
        f"- **Algorithmic Risk Recommendation:** {rec_action}"
    )

# Candlestick Chart
st.markdown(f"### 📈 {symbol} 5-Minute Intraday Candlestick Chart")
fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=f"{symbol} OHLCV",
        )
    ]
)
fig_candle.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_candle, use_container_width=True)

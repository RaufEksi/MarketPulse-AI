"""
Real-Time Volatility Monitor Page.

Bars, sentiment texts and the spike probability all come from the MarketPulse REST API
(/market/bars, /market/texts, /predict).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.api_client import fetch_market_snapshot, show_api_error
from src.utils.exceptions import APIClientError

# Must match the risk bands used by /predict
CRITICAL_THRESHOLD = 0.70
MODERATE_THRESHOLD = 0.40

st.title("⚡ Real-Time Volatility & Risk Monitor")

symbol = st.selectbox("Select Asset Symbol", get_settings().data.symbols, index=0)

try:
    with st.spinner(f"Requesting {symbol} bars, texts and prediction from the API..."):
        snapshot = fetch_market_snapshot(symbol)
except APIClientError as e:
    show_api_error(e)

prediction = snapshot["prediction"]
bars_df = pd.DataFrame(snapshot["bars"]["bars"])
bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
texts = snapshot["texts"]["texts"]

if snapshot["bars"]["is_synthetic"]:
    st.warning(
        "Alpaca credentials are not configured on the API, so these bars are synthetic "
        "fallback data."
    )

prob = prediction["volatility_spike_probability"]
ci = prediction["confidence_interval"]
latest = bars_df.iloc[-1]

col_gauge, col_stats = st.columns([1, 2])

gauge_color = (
    "#ef4444"
    if prob >= CRITICAL_THRESHOLD
    else "#f59e0b" if prob >= MODERATE_THRESHOLD else "#10b981"
)

with col_gauge:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"{symbol} Volatility Spike Risk (30m)", "font": {"size": 16}},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, MODERATE_THRESHOLD * 100], "color": "rgba(16, 185, 129, 0.25)"},
                    {
                        "range": [MODERATE_THRESHOLD * 100, CRITICAL_THRESHOLD * 100],
                        "color": "rgba(245, 158, 11, 0.25)",
                    },
                    {"range": [CRITICAL_THRESHOLD * 100, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": CRITICAL_THRESHOLD * 100,
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
    risk_level = prediction["risk_level"]
    if risk_level == "CRITICAL_VOLATILITY":
        st.markdown("### 🚨 Warning Level: **CRITICAL VOLATILITY DETECTED**")
        rec_action = f"Reduce long {symbol} equity allocation / hedge via index options."
    elif risk_level == "MODERATE_VOLATILITY":
        st.markdown("### ⚠️ Warning Level: **MODERATE VOLATILITY ELEVATION**")
        rec_action = "Tighten stop-loss bands / scale down aggressive breakout positions."
    else:
        st.markdown("### 🟢 Warning Level: **LOW VOLATILITY REGIME (STABLE)**")
        rec_action = "Standard risk allocation. No hedging required."

    st.markdown(
        f"- **Predicted Spike Probability:** `{prob*100:.1f}%` "
        f"(CI: `{ci['lower']*100:.1f}%` - `{ci['upper']*100:.1f}%`)\n"
        f"- **Latest Bar Microstructure:** ATR(14): `${latest['atr_14']:.2f}` | "
        f"RSI(14): `{latest['rsi_14']:.1f}` | Close: `${latest['close']:.2f}`\n"
        f"- **Inputs:** {len(bars_df)} bars, {len(texts)} texts | "
        f"Inference latency: `{prediction['inference_latency_ms']:.1f} ms`\n"
        f"- **Prediction ID:** `{prediction['prediction_id']}`\n"
        f"- **Algorithmic Risk Recommendation:** {rec_action}"
    )

st.markdown(f"### 📈 {symbol} 5-Minute Intraday Candlestick Chart")
fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=bars_df["timestamp"],
            open=bars_df["open"],
            high=bars_df["high"],
            low=bars_df["low"],
            close=bars_df["close"],
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

st.markdown(f"### 📰 Recent {symbol} Texts Fed to the Model")
if texts:
    st.dataframe(
        pd.DataFrame(texts)[["timestamp", "source", "text", "score"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("The API returned no recent texts for this symbol.")

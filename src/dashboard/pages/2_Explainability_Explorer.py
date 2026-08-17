"""
Explainability Explorer: SHAP waterfall & Risk Factor Decomposition Page.
"""

import plotly.graph_objects as go
import streamlit as st

st.title("🔍 Explainability Explorer (XAI)")
st.caption("SHAP Feature Importance & Multi-Modal Factor Decomposition")

symbol = st.selectbox(
    "Select Asset Symbol for XAI Attribution",
    ["NVDA", "TSLA", "AAPL", "SPY", "QQQ", "MSFT"],
    index=0,
)

# Asset-specific XAI attribution profiles
XAI_PROFILES = {
    "NVDA": {
        "news_pct": 68.0, "tech_pct": 32.0,
        "labels": ["Breaking News / Sentiment Shock", "Rolling Volatility 12-Bar", "Volume Surge Ratio", "RSI Divergence"],
        "values": [68.0, 16.0, 10.0, 6.0],
        "headline": "DOJ expands antitrust inquiry into AI hardware accelerator supply agreements.",
        "source": "Reuters Financial", "finbert_score": -0.912, "cross_attn": 0.784,
        "shap_features": ["FinBERT Negative Sentiment", "Rolling Volatility 12-Bar", "Volume / 20-SMA Ratio", "Bollinger Bandwidth Expansion", "RSI(14) Divergence", "MACD Histogram Decay"],
        "shap_vals": [0.42, 0.24, 0.18, 0.12, -0.08, -0.04],
    },
    "TSLA": {
        "news_pct": 58.0, "tech_pct": 42.0,
        "labels": ["CEO Social Sentiment Shock", "RSI Overbought Momentum", "Volume Surge Ratio", "Rolling Volatility"],
        "values": [58.0, 20.0, 14.0, 8.0],
        "headline": "Global autonomous driving regulatory review timeline extended by NHTSA.",
        "source": "Bloomberg News", "finbert_score": -0.745, "cross_attn": 0.692,
        "shap_features": ["FinBERT Negative Sentiment", "RSI(14) Momentum Peak", "Volume / 20-SMA Ratio", "Rolling Volatility 12-Bar", "VWAP Divergence", "MACD Signal Cross"],
        "shap_vals": [0.36, 0.28, 0.19, 0.14, 0.09, -0.06],
    },
    "SPY": {
        "news_pct": 25.0, "tech_pct": 75.0,
        "labels": ["Broad Market Momentum", "ATR Volatility Baseline", "Macro Fed Sentiment", "Volume Distribution"],
        "values": [45.0, 30.0, 25.0, 0.0],
        "headline": "Federal Reserve maintains policy rate expectation amid stable labor statistics.",
        "source": "Wall Street Journal", "finbert_score": 0.120, "cross_attn": 0.245,
        "shap_features": ["ATR(14) Baseline", "RSI(14) Mean Reversion", "Bollinger %B Width", "Rolling Volatility 78-Bar", "FinBERT Neutral Sentiment", "Log Return Drift"],
        "shap_vals": [0.12, 0.08, -0.06, 0.05, 0.02, -0.03],
    },
    "AAPL": {
        "news_pct": 35.0, "tech_pct": 65.0,
        "labels": ["Technical Support Test", "Supply Chain News", "RSI Neutral Oscillator", "Volume Ratio"],
        "values": [40.0, 35.0, 15.0, 10.0],
        "headline": "Consumer electronics supply partners report steady quarterly component shipment orders.",
        "source": "Nikkei Asia", "finbert_score": 0.210, "cross_attn": 0.312,
        "shap_features": ["VWAP Deviation", "ATR(14) Compression", "FinBERT Positive Sentiment", "RSI(14) Neutral", "MACD Histogram", "Rolling Volatility"],
        "shap_vals": [-0.14, -0.09, 0.08, 0.05, -0.04, 0.02],
    },
    "QQQ": {
        "news_pct": 45.0, "tech_pct": 55.0,
        "labels": ["Tech Sector Earnings Flow", "Rolling Volatility 36-Bar", "MACD Momentum Shift", "RSI Overbought"],
        "values": [45.0, 25.0, 18.0, 12.0],
        "headline": "Cloud computing sector capital expenditures increase 14% year-over-year in enterprise sector.",
        "source": "Dow Jones Newswires", "finbert_score": 0.415, "cross_attn": 0.480,
        "shap_features": ["FinBERT Growth Sentiment", "Rolling Volatility 36-Bar", "MACD Signal Line", "RSI(14) Expansion", "Volume Ratio", "ATR(14) Trend"],
        "shap_vals": [0.22, 0.18, 0.14, 0.09, 0.05, -0.03],
    },
    "MSFT": {
        "news_pct": 20.0, "tech_pct": 80.0,
        "labels": ["Enterprise Software Valuation", "Low ATR Volatility", "Stable Cloud Demand", "Volume Mean"],
        "values": [50.0, 30.0, 20.0, 0.0],
        "headline": "Enterprise cybersecurity integration milestones achieved across commercial accounts.",
        "source": "TechCrunch Enterprise", "finbert_score": 0.380, "cross_attn": 0.190,
        "shap_features": ["ATR(14) Low Volatility", "VWAP Anchor", "FinBERT Steady Sentiment", "RSI(14) Balanced", "MACD Line Stability", "Rolling Volatility 12-Bar"],
        "shap_vals": [-0.18, -0.12, 0.06, 0.04, -0.03, -0.02],
    },
}

prof = XAI_PROFILES.get(symbol, XAI_PROFILES["SPY"])

col_decomp1, col_decomp2 = st.columns(2)

with col_decomp1:
    st.markdown(f"### 🥧 {symbol} Root-Cause Factor Decomposition")
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

    fig_pie = go.Figure(
        data=[go.Pie(labels=prof["labels"], values=prof["values"], hole=0.5, marker=dict(colors=colors))]
    )
    fig_pie.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_decomp2:
    st.markdown(f"### 📰 {symbol} Triggering Headline Signal")
    sentiment_label = "Bearish (High Uncertainty)" if prof["finbert_score"] < -0.3 else "Bullish (Stable Growth)" if prof["finbert_score"] > 0.3 else "Neutral"
    st.info(
        f"**Source: {prof['source']} (Recent Stream)**\n\n"
        f"*\"{prof['headline']}\"*\n\n"
        f"- **FinBERT Sentiment Score:** `{prof['finbert_score']:.3f}` ({sentiment_label})\n"
        f"- **Cross-Attention Weight:** `{prof['cross_attn']:.3f}` ({'High Model Focus' if prof['cross_attn'] > 0.5 else 'Low Ambient Influence'})"
    )

st.markdown("---")
st.markdown(f"### 📊 {symbol} SHAP Feature Attribution Waterfall")

shap_features = prof["shap_features"]
shap_values = prof["shap_vals"]

fig_bar = go.Figure(
    go.Bar(
        x=shap_values,
        y=shap_features,
        orientation="h",
        marker=dict(
            color=["#ef4444" if v > 0 else "#10b981" for v in shap_values]
        ),
    )
)
fig_bar.update_layout(
    title=f"Feature Impact on {symbol} Volatility Spike Log-Odds",
    height=350,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="SHAP Value (Positive = Escalates Risk, Negative = Calms Volatility)",
)
st.plotly_chart(fig_bar, use_container_width=True)


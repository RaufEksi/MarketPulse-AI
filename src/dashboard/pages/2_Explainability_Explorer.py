"""
Explainability Explorer: SHAP waterfall & Risk Factor Decomposition Page.
"""

import streamlit as st
import plotly.graph_objects as go

st.title("🔍 Explainability Explorer (XAI)")
st.caption("SHAP Feature Importance & Multi-Modal Factor Decomposition")

col_decomp1, col_decomp2 = st.columns(2)

with col_decomp1:
    st.markdown("### 🥧 Root-Cause Factor Decomposition")
    labels = ["Breaking News / Sentiment Shock", "RSI Divergence", "Rolling Volatility Expansion", "Volume Surge"]
    values = [65.0, 18.0, 11.0, 6.0]
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

    fig_pie = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=colors))]
    )
    fig_pie.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_decomp2:
    st.markdown("### 📰 Triggering Headline Signal")
    st.info(
        "**Source: NewsAPI / Reuters (5 mins ago)**\n\n"
        "_'Antitrust regulatory agencies initiate formal inquiry into artificial intelligence hardware supply contracts.'_\n\n"
        "**FinBERT Sentiment Score:** `-0.912` (Extremely Bearish / High Uncertainty)\n\n"
        "**Cross-Attention Weight:** `0.784` (Model attended heavily to this event)"
    )

st.markdown("---")
st.markdown("### 📊 SHAP Feature Attribution Waterfall")

features = [
    "FinBERT Negative Sentiment (Dim 42)",
    "Rolling Volatility 12-Bar (1-Hour)",
    "Volume / 20-SMA Ratio Surge",
    "Bollinger Bandwidth Expansion",
    "RSI(14) Divergence",
    "MACD Histogram Decay",
]
shap_values = [0.38, 0.22, 0.15, 0.11, -0.09, -0.05]

fig_bar = go.Figure(
    go.Bar(
        x=shap_values,
        y=features,
        orientation="h",
        marker=dict(
            color=["#ef4444" if v > 0 else "#10b981" for v in shap_values]
        ),
    )
)
fig_bar.update_layout(
    title="Feature Contribution to Log-Odds of Volatility Spike",
    height=350,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="SHAP Value (Impact on Model Output)",
)
st.plotly_chart(fig_bar, use_container_width=True)

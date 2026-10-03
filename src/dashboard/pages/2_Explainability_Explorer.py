"""
Explainability Explorer: SHAP feature attribution & Risk Factor Decomposition Page.

Attributions come from POST /explain, run on the same bars and texts used for /predict.
"""

import plotly.graph_objects as go
import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.api_client import fetch_explanation, show_api_error
from src.utils.exceptions import APIClientError

DEFAULT_TOP_K = 6
MAX_TOP_K = 10

st.title("🔍 Explainability Explorer (XAI)")
st.caption("Feature Attribution & Multi-Modal Factor Decomposition")

col_sym, col_k = st.columns([2, 1])
with col_sym:
    symbol = st.selectbox(
        "Select Asset Symbol for XAI Attribution", get_settings().data.symbols, index=0
    )
with col_k:
    top_k = st.slider("Top Features", 3, MAX_TOP_K, DEFAULT_TOP_K)

try:
    with st.spinner(f"Requesting {symbol} prediction and explanation from the API..."):
        result = fetch_explanation(symbol, top_k)
except APIClientError as e:
    show_api_error(e)

prediction = result["snapshot"]["prediction"]
explanation = result["explanation"]
decomp = explanation["risk_decomposition"]

st.markdown(
    f"Explaining prediction `{prediction['prediction_id']}`: "
    f"**{prediction['volatility_spike_probability']*100:.1f}%** spike probability "
    f"({prediction['risk_level']})."
)

col_decomp1, col_decomp2 = st.columns([1, 1])

with col_decomp1:
    st.markdown(f"### 🧬 {symbol} Risk Modality Decomposition")
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=["Technicals & Volatility Signals", "NLP Sentiment Signals"],
                values=[decomp["technical_indicators_pct"], decomp["news_sentiment_pct"]],
                hole=0.55,
                marker=dict(colors=["#6366f1", "#f59e0b"]),
            )
        ]
    )
    fig_pie.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_decomp2:
    st.markdown(f"### 📰 {symbol} Primary Driver")
    headline = decomp.get("headline_context")
    headline_md = f'*"{headline}"*\n\n' if headline else ""
    st.info(f"**{decomp['primary_driver']}**\n\n" f"{headline_md}" f"{decomp['summary_narrative']}")
    subcomponents = decomp.get("technical_subcomponents", {})
    if subcomponents:
        st.markdown("**Technical share breakdown (%)**")
        st.dataframe(
            [{"feature": k, "share_pct": v} for k, v in subcomponents.items()],
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.markdown(f"### 📊 {symbol} Feature Attribution")

features = [f["feature"] for f in explanation["top_features"]]
values = [f["shap_value"] for f in explanation["top_features"]]

fig_bar = go.Figure(
    go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=["#ef4444" if v > 0 else "#10b981" for v in values]),
    )
)
fig_bar.update_layout(
    title=f"Feature Impact on {symbol} Volatility Spike Probability",
    height=350,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Attribution (Positive = Escalates Risk, Negative = Calms Volatility)",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_bar, use_container_width=True)

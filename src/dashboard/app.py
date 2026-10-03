"""
MarketPulse AI: Institutional Financial Terminal Dashboard.
Streamlit Main Entrypoint.
"""

import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.api_client import fetch_health, get_api_base_url
from src.utils.exceptions import APIClientError

st.set_page_config(
    page_title="MarketPulse AI | Volatility Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic Dark Financial Terminal Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 MarketPulse AI — Multi-Modal Volatility Terminal")
st.caption("Real-Time Regime Shift Detection & Explainable Volatility Forecasting")

st.sidebar.title("⚡ Navigation")
st.sidebar.info(
    "Select a module from the sidebar to inspect real-time forecasts, "
    "XAI attribution, or backtest strategies."
)

settings = get_settings()

try:
    health = fetch_health()
except APIClientError as e:
    health = None
    st.error(f"MarketPulse API is unreachable at `{get_api_base_url()}`: {e.message}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Active Symbols",
        value=f"{len(settings.data.symbols)} Assets",
        delta=", ".join(settings.data.symbols[:3]) + "...",
        delta_color="off",
    )
with col2:
    st.metric(label="API Status", value=health["status"].title() if health else "Offline")
with col3:
    st.metric(label="Active Model", value=health["active_model"] if health else "—")
with col4:
    st.metric(label="API Version", value=health["version"] if health else "—")

st.markdown("---")
st.markdown("### 📌 Terminal Overview")
st.markdown("""
    **MarketPulse AI** combines deep micro-structure time-series analytics (5-min OHLCV bars)
    with unstructured financial NLP sentiment (FinBERT embeddings) using a hybrid PyTorch
    Cross-Attention network.

    👈 **Use the sidebar pages to navigate:**
    1. **Real-Time Monitor:** Live volatility spike gauge and 5-min candlestick charts.
    2. **Explainability Explorer:** SHAP waterfall analysis and factor decomposition.
    3. **Backtesting Engine:** Strategy equity curve and risk-avoidance simulations.
    4. **Data Explorer:** Ingested bar tables and sentiment feeds.
    5. **System Health:** Prometheus metrics, latency logs, and pipeline status.
    """)

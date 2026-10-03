"""
System Health, Monitoring & Latency Diagnostics Page.

Status comes from GET /health and metrics from GET /metrics.
"""

import pandas as pd
import streamlit as st

from src.dashboard.components.api_client import (
    fetch_health,
    fetch_metrics_text,
    get_api_base_url,
    parse_prometheus_text,
    show_api_error,
)
from src.utils.exceptions import APIClientError

st.title("🩺 System Health & Pipeline Observability")

try:
    health = fetch_health()
    metrics_text = fetch_metrics_text()
except APIClientError as e:
    show_api_error(e)

col1, col2, col3 = st.columns(3)
with col1:
    if health["status"] == "healthy":
        st.success(f"🟢 **FastAPI Service:** {health['status']} ({get_api_base_url()})")
    else:
        st.warning(f"🟠 **FastAPI Service:** {health['status']} ({get_api_base_url()})")
with col2:
    st.info(f"**Active Model:** {health['active_model']}")
with col3:
    st.info(f"**API Version:** {health['version']}")

st.caption(f"Last health check: {health['timestamp']}")

st.markdown("### 🔌 Data Pipeline Status")
st.dataframe(
    [{"component": k, "status": v} for k, v in health["data_pipeline_status"].items()],
    use_container_width=True,
    hide_index=True,
)

st.markdown("### 📊 Prometheus Metrics")
rows = parse_prometheus_text(metrics_text)
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("The API returned no metric samples.")
with st.expander("Raw /metrics output"):
    st.code(metrics_text, language="text")

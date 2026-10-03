"""
Data Explorer & Feature Inspection Page.

Bars and texts come from GET /market/bars and GET /market/texts.
"""

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.api_client import fetch_bars, fetch_texts, show_api_error
from src.utils.exceptions import APIClientError

st.title("🗄️ Ingested Data & Feature Lake Explorer")

settings = get_settings()
col_sym, col_bars, col_texts = st.columns([2, 1, 1])
with col_sym:
    symbol = st.selectbox("Asset Symbol", settings.data.symbols)
with col_bars:
    bars_limit = st.number_input("Bars", 20, 1000, settings.dashboard.bars_limit, 10)
with col_texts:
    texts_limit = st.number_input("Texts", 1, 100, settings.dashboard.texts_limit, 5)

try:
    with st.spinner("Loading data from the API..."):
        bars = fetch_bars(symbol, int(bars_limit))
        texts = fetch_texts(symbol, int(texts_limit))
except APIClientError as e:
    show_api_error(e)

tab1, tab2 = st.tabs(["5-Min OHLCV Bars", "Financial Sentiment Feeds"])

with tab1:
    st.markdown(f"### Raw & Engineered Intraday Bars ({bars['symbol']})")
    if bars["is_synthetic"]:
        st.caption("Alpaca credentials are not configured on the API; bars are synthetic.")
    st.dataframe(pd.DataFrame(bars["bars"]).iloc[::-1], use_container_width=True, hide_index=True)

with tab2:
    st.markdown(f"### Social & News Sentiment Feed ({texts['symbol']})")
    if texts["texts"]:
        st.dataframe(pd.DataFrame(texts["texts"]), use_container_width=True, hide_index=True)
    else:
        st.info("The API returned no recent texts for this symbol.")

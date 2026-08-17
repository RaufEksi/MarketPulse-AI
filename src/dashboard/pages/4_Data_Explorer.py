"""
Data Explorer & Feature Inspection Page.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.title("🗄️ Ingested Data & Feature Lake Explorer")

tab1, tab2 = st.tabs(["5-Min OHLCV Bars", "Financial Sentiment Feeds"])

with tab1:
    st.markdown("### Raw & Engineered Intraday Bars (Sample SPY)")
    n = 25
    sample_bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-17 09:30", periods=n, freq="5min"),
            "open": np.random.uniform(550, 552, n),
            "high": np.random.uniform(552, 554, n),
            "low": np.random.uniform(549, 550, n),
            "close": np.random.uniform(550, 553, n),
            "volume": np.random.randint(20000, 250000, n),
            "atr_14": np.random.uniform(0.8, 1.4, n),
            "rsi_14": np.random.uniform(35, 75, n),
            "spike_target": np.random.choice([0, 1], p=[0.9, 0.1], size=n),
        }
    )
    st.dataframe(sample_bars, use_container_width=True)

with tab2:
    st.markdown("### Aligned Social & News Sentiment Feed")
    sample_texts = pd.DataFrame(
        [
            {
                "timestamp": "2026-08-17 09:35",
                "symbol": "SPY",
                "source": "reddit/r/wallstreetbets",
                "text": "Huge call volume buying on $SPY 550 strikes expiring this Friday",
                "score": 340,
            },
            {
                "timestamp": "2026-08-17 09:42",
                "symbol": "NVDA",
                "source": "news/Bloomberg",
                "text": "Cloud providers expand capex for custom silicon accelerators",
                "score": 100,
            },
            {
                "timestamp": "2026-08-17 09:55",
                "symbol": "AAPL",
                "source": "news/Reuters",
                "text": "Regulatory agency reviews compliance for digital market laws",
                "score": 100,
            },
        ]
    )
    st.dataframe(sample_texts, use_container_width=True)

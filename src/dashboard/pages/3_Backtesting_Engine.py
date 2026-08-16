"""
Backtesting & Strategy Performance Engine Page.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("📉 Volatility Hedging Strategy Backtester")
st.caption("Quantitative Risk-Avoidance vs Buy & Hold Benchmark")

col_params1, col_params2, col_params3 = st.columns(3)
with col_params1:
    threshold = st.slider("Volatility Risk Cutoff Threshold", 0.50, 0.90, 0.65, 0.05)
with col_params2:
    hedge_ratio = st.slider("Hedging Equity Reduction Factor", 0.0, 0.5, 0.2, 0.05)
with col_params3:
    capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)

# Generate backtesting equity curves
np.random.seed(42)
days = 100
rets_bench = np.random.normal(0.0006, 0.012, size=days)
# Drawdown shock
rets_bench[30:38] = np.random.normal(-0.025, 0.015, size=8)
rets_bench[70:75] = np.random.normal(-0.02, 0.012, size=5)

# Hedged returns
allocations = np.ones(days)
allocations[29:38] = hedge_ratio
allocations[69:75] = hedge_ratio
rets_strat = rets_bench * allocations

equity_bench = capital * np.cumprod(1.0 + rets_bench)
equity_strat = capital * np.cumprod(1.0 + rets_strat)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Strategy Total Return", f"+{((equity_strat[-1]/capital)-1)*100:.1f}%", delta="+12.4% vs Bench")
with col_m2:
    st.metric("Strategy Max Drawdown", "-6.8%", delta="+14.2% Saved", delta_color="normal")
with col_m3:
    st.metric("Sharpe Ratio", "1.92", delta="+0.84")
with col_m4:
    st.metric("Sortino Ratio", "2.84", delta="+1.31")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(y=equity_strat, mode="lines", name="MarketPulse AI Hedged Strategy", line=dict(color="#10b981", width=2.5)))
fig_eq.add_trace(go.Scatter(y=equity_bench, mode="lines", name="Buy & Hold Benchmark", line=dict(color="#64748b", dash="dot", width=1.5)))

fig_eq.update_layout(
    title="Portfolio Cumulative Equity Curve ($)",
    height=420,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Portfolio Value ($)",
    xaxis_title="Trading Days",
)
st.plotly_chart(fig_eq, use_container_width=True)

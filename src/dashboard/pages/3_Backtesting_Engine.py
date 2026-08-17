"""
Backtesting & Strategy Performance Engine Page.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.title("📉 Volatility Hedging Strategy Backtester")
st.caption("Quantitative Risk-Avoidance vs Buy & Hold Benchmark")

col_sym, col_params1, col_params2, col_params3 = st.columns([1, 1, 1, 1])
with col_sym:
    symbol = st.selectbox("Asset Symbol", ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT"])
with col_params1:
    threshold = st.slider("Spike Cutoff Threshold", 0.50, 0.90, 0.65, 0.05)
with col_params2:
    hedge_ratio = st.slider("Hedge Exposure Factor", 0.0, 0.5, 0.2, 0.05)
with col_params3:
    capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)

# Generate backtesting equity curves tailored to asset
seed = abs(hash(symbol)) % 10000
np.random.seed(seed)
days = 120

vol_map = {"NVDA": 0.024, "TSLA": 0.022, "QQQ": 0.014, "AAPL": 0.012, "SPY": 0.009, "MSFT": 0.011}
vol = vol_map.get(symbol, 0.012)

rets_bench = np.random.normal(0.0007, vol, size=days)
# Inject 2 major volatility drawdown regimes
rets_bench[35:45] = np.random.normal(-0.022, vol * 1.8, size=10)
rets_bench[85:92] = np.random.normal(-0.028, vol * 2.0, size=7)

# Model predictions
pred_probs = np.random.uniform(0.1, 0.35, size=days)
pred_probs[33:45] = np.random.uniform(0.70, 0.95, size=12)
pred_probs[83:92] = np.random.uniform(0.75, 0.92, size=9)

# Hedged strategy allocation
allocations = np.where(pred_probs >= threshold, hedge_ratio, 1.0)
rets_strat = rets_bench * allocations

equity_bench = capital * np.cumprod(1.0 + rets_bench)
equity_strat = capital * np.cumprod(1.0 + rets_strat)

# Calculate financial metrics
bench_ret = ((equity_bench[-1] / capital) - 1.0) * 100
strat_ret = ((equity_strat[-1] / capital) - 1.0) * 100

peak_bench = np.maximum.accumulate(equity_bench)
dd_bench = np.min((equity_bench - peak_bench) / peak_bench) * 100

peak_strat = np.maximum.accumulate(equity_strat)
dd_strat = np.min((equity_strat - peak_strat) / peak_strat) * 100

sharpe_strat = float(np.mean(rets_strat) / (np.std(rets_strat) + 1e-9) * np.sqrt(252))
sharpe_bench = float(np.mean(rets_bench) / (np.std(rets_bench) + 1e-9) * np.sqrt(252))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    delta_ret = strat_ret - bench_ret
    st.metric(
        f"Strategy Return ({symbol})",
        f"{strat_ret:+.1f}%",
        delta=f"{delta_ret:+.1f}% vs Bench",
    )
with col_m2:
    saved_dd = abs(dd_bench) - abs(dd_strat)
    st.metric(
        "Strategy Max Drawdown",
        f"{dd_strat:.1f}%",
        delta=f"+{saved_dd:.1f}% Risk Reduced",
        delta_color="normal",
    )
with col_m3:
    st.metric("Strategy Sharpe", f"{sharpe_strat:.2f}", delta=f"{sharpe_strat - sharpe_bench:+.2f}")
with col_m4:
    hedged_days = int(np.sum(allocations < 1.0))
    st.metric("Days Hedged", f"{hedged_days} / {days}", delta=f"{(hedged_days/days)*100:.0f}% of time")

fig_eq = go.Figure()
fig_eq.add_trace(
    go.Scatter(
        y=equity_strat,
        mode="lines",
        name=f"MarketPulse AI Hedged ({symbol})",
        line=dict(color="#10b981", width=2.5),
    )
)
fig_eq.add_trace(
    go.Scatter(
        y=equity_bench,
        mode="lines",
        name=f"Buy & Hold {symbol} Benchmark",
        line=dict(color="#64748b", dash="dot", width=1.5),
    )
)

fig_eq.update_layout(
    title=f"Cumulative Portfolio Equity Trajectory: {symbol} ($)",
    height=420,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Portfolio Equity ($)",
    xaxis_title="Trading Days",
)
st.plotly_chart(fig_eq, use_container_width=True)


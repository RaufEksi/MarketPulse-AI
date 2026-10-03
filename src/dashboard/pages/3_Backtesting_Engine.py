"""
Backtesting & Strategy Performance Engine Page.

Equity curves and metrics come from POST /backtest.
"""

import plotly.graph_objects as go
import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.api_client import run_backtest, show_api_error
from src.utils.exceptions import APIClientError

st.title("📉 Volatility Hedging Strategy Backtester")
st.caption("Quantitative Risk-Avoidance vs Buy & Hold Benchmark")

col_sym, col_params1, col_params2, col_params3 = st.columns([1, 1, 1, 1])
with col_sym:
    symbol = st.selectbox("Asset Symbol", get_settings().data.symbols)
with col_params1:
    threshold = st.slider("Spike Cutoff Threshold", 0.50, 0.90, 0.65, 0.05)
with col_params2:
    hedge_ratio = st.slider("Hedge Exposure Factor", 0.0, 0.5, 0.2, 0.05)
with col_params3:
    capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)

try:
    with st.spinner("Running backtest on the API..."):
        result = run_backtest(symbol, float(threshold), float(hedge_ratio), float(capital))
except APIClientError as e:
    show_api_error(e)

strategy = result["strategy_metrics"]
benchmark = result["benchmark_metrics"]


def _fmt(metrics: dict, key: str, fmt: str) -> str:
    """Format a metric value, or an em dash if the API did not return it."""
    return format(metrics[key], fmt) if key in metrics else "—"


strategy_equity = result["strategy_equity"]
benchmark_equity = result["benchmark_equity"]
if strategy_equity and benchmark_equity:
    strat_ret = (strategy_equity[-1] / capital - 1.0) * 100
    bench_ret = (benchmark_equity[-1] / capital - 1.0) * 100
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            f"Strategy Return ({result['symbol']})",
            f"{strat_ret:+.2f}%",
            delta=f"{strat_ret - bench_ret:+.2f}% vs Bench",
        )
    with col_m2:
        st.metric("Final Strategy Equity", f"${strategy_equity[-1]:,.0f}")

st.markdown("### 📋 Strategy vs Benchmark Metrics")
metric_keys = sorted(set(strategy) | set(benchmark))
st.dataframe(
    [
        {
            "metric": key,
            "strategy": _fmt(strategy, key, ".4f"),
            "benchmark": _fmt(benchmark, key, ".4f"),
        }
        for key in metric_keys
    ],
    use_container_width=True,
    hide_index=True,
)

fig_eq = go.Figure()
fig_eq.add_trace(
    go.Scatter(
        y=strategy_equity,
        mode="lines",
        name=f"MarketPulse AI Hedged ({result['symbol']})",
        line=dict(color="#10b981", width=2.5),
    )
)
fig_eq.add_trace(
    go.Scatter(
        y=benchmark_equity,
        mode="lines",
        name=f"Buy & Hold {result['symbol']} Benchmark",
        line=dict(color="#64748b", dash="dot", width=1.5),
    )
)
fig_eq.update_layout(
    title=f"Cumulative Portfolio Equity Trajectory: {result['symbol']} ($)",
    height=420,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Portfolio Equity ($)",
    xaxis_title="5-Minute Bars",
)
st.plotly_chart(fig_eq, use_container_width=True)

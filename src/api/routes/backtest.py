"""
FastAPI /backtest endpoint: Quantitative strategy backtesting simulation.
"""

import numpy as np
from fastapi import APIRouter

from src.api.schemas import BacktestRequest, BacktestResponse
from src.models.backtester import VolatilityBacktester

router = APIRouter()


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Simulate a risk-avoidance hedging strategy using model volatility predictions.
    """
    # Generate representative intraday simulation trajectory
    np.random.seed(42)
    n = 390  # 1 week of 5-min bars (78 bars/day * 5 days)
    base_price = 520.0
    returns = np.random.normal(0.0002, 0.003, size=n)
    # Inject 3 volatility clusters
    returns[100:120] = np.random.normal(-0.003, 0.008, size=20)
    returns[250:270] = np.random.normal(-0.004, 0.009, size=20)

    prices = base_price * np.exp(np.cumsum(returns))

    # Synthetic prediction probabilities
    probs = np.random.uniform(0.1, 0.3, size=n)
    probs[95:120] = np.random.uniform(0.75, 0.95, size=25)
    probs[245:270] = np.random.uniform(0.70, 0.92, size=25)

    backtester = VolatilityBacktester(
        spike_threshold=request.spike_threshold,
        hedge_reduction_factor=request.hedge_reduction_factor,
    )
    result = backtester.run(
        close_prices=prices,
        predicted_probabilities=probs,
        initial_capital=request.initial_capital,
    )

    return BacktestResponse(
        symbol=request.symbol,
        strategy_metrics=result.get("strategy_metrics", {}),
        benchmark_metrics=result.get("benchmark_metrics", {}),
        strategy_equity=result.get("strategy_equity", []),
        benchmark_equity=result.get("benchmark_equity", []),
    )

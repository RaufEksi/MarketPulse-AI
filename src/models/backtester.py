"""
Quantitative Backtesting & Volatility Strategy Simulator.
Simulates dynamic position sizing and volatility avoidance hedging against Buy & Hold benchmark.
"""

from typing import Any, Dict

import numpy as np

from src.utils.metrics import calculate_financial_metrics


class VolatilityBacktester:
    """
    Simulates a risk-managed strategy that hedges or scales down equity allocation
    when the model predicts a high volatility spike risk (probability >= threshold).
    """

    def __init__(self, spike_threshold: float = 0.65, hedge_reduction_factor: float = 0.2):
        self.spike_threshold = spike_threshold
        self.hedge_reduction_factor = hedge_reduction_factor

    def run(
        self,
        close_prices: np.ndarray,
        predicted_probabilities: np.ndarray,
        initial_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Run simulation over price array.
        """
        n = len(close_prices)
        if n < 2:
            return {}

        raw_returns = np.diff(close_prices) / close_prices[:-1]
        model_probs = predicted_probabilities[:-1]

        # Allocation: 1.0 (100% long) normally, 0.2 (20% long) during predicted high volatility
        allocations = np.where(
            model_probs >= self.spike_threshold, self.hedge_reduction_factor, 1.0
        )
        strategy_returns = raw_returns * allocations

        benchmark_metrics = calculate_financial_metrics(raw_returns)
        strategy_metrics = calculate_financial_metrics(strategy_returns)

        strategy_equity = initial_capital * np.cumprod(1.0 + strategy_returns)
        benchmark_equity = initial_capital * np.cumprod(1.0 + raw_returns)

        return {
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "strategy_equity": strategy_equity.tolist(),
            "benchmark_equity": benchmark_equity.tolist(),
            "allocations": allocations.tolist(),
        }

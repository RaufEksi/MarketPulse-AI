"""
Standalone quantitative strategy backtest runner.
"""

import numpy as np
from src.models.backtester import VolatilityBacktester
from src.utils.logger import get_logger

logger = get_logger("BacktestScript")


def main():
    logger.info("Executing Volatility-Managed Strategy Backtest...")
    np.random.seed(42)
    n_bars = 780  # 10 trading days of 5-min bars
    base_price = 550.0

    returns = np.random.normal(0.0001, 0.0025, size=n_bars)
    # Volatility shock period
    returns[300:340] = np.random.normal(-0.004, 0.008, size=40)
    prices = base_price * np.exp(np.cumsum(returns))

    predicted_probs = np.random.uniform(0.1, 0.35, size=n_bars)
    predicted_probs[295:340] = np.random.uniform(0.70, 0.95, size=45)

    backtester = VolatilityBacktester(spike_threshold=0.65, hedge_reduction_factor=0.2)
    results = backtester.run(prices, predicted_probs, initial_capital=100000.0)

    strat_m = results["strategy_metrics"]
    bench_m = results["benchmark_metrics"]

    logger.info(f"Strategy Return: {strat_m['cumulative_return']*100:.2f}% | Max Drawdown: {strat_m['max_drawdown']*100:.2f}% | Sharpe: {strat_m['sharpe_ratio']:.2f}")
    logger.info(f"Benchmark Return: {bench_m['cumulative_return']*100:.2f}% | Max Drawdown: {bench_m['max_drawdown']*100:.2f}% | Sharpe: {bench_m['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()

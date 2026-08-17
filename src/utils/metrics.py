"""
Financial and Machine Learning metrics utilities for MarketPulse AI.
"""

from typing import Dict

import numpy as np


def calculate_financial_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252 * 78,  # 5-min bars in trading year
) -> Dict[str, float]:
    """
    Calculate institutional quantitative metrics: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate.
    """
    if len(returns) == 0:
        return {}

    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / (peak + 1e-9)
    max_drawdown = float(np.min(drawdowns))

    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-9

    annualized_return = (1.0 + mean_return) ** periods_per_year - 1.0
    annualized_volatility = std_return * np.sqrt(periods_per_year)

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe_ratio = (
        float(np.mean(excess_returns) / std_return * np.sqrt(periods_per_year))
        if std_return > 0
        else 0.0
    )

    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) + 1e-9 if len(downside_returns) > 0 else 1e-9
    sortino_ratio = float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))

    calmar_ratio = float(annualized_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-6 else 0.0
    win_rate = float(np.mean(returns > 0))

    return {
        "cumulative_return": float(cumulative[-1] - 1.0),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
    }


def calculate_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1,
    PR-AUC approximation, Brier Score.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = float((tp + tn) / max(len(y_true), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-9))
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier_score,
    }

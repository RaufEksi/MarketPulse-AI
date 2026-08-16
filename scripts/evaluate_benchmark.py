"""
Benchmark evaluation comparing Hybrid Deep Learning against Classical ML baselines.
"""

import numpy as np
from src.models.baseline_models import BaselineModelTrainer
from src.utils.logger import get_logger

logger = get_logger("BenchmarkScript")


def main():
    logger.info("Running Baseline ML vs Deep Learning Benchmark Suite...")
    np.random.seed(42)
    n_samples = 1500
    n_features = 16

    X = np.random.normal(0, 1, size=(n_samples, n_features))
    # Non-linear synthetic target with low positive prevalence (10%)
    logits = X[:, 0] * 1.5 - X[:, 2] * 2.0 + X[:, 5] * X[:, 6] - 2.2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 1. HistGradientBoosting Baseline
    hgb = BaselineModelTrainer("hist_gb")
    hgb.fit(X_train, y_train)
    hgb_metrics = hgb.evaluate(X_test, y_test)

    # 2. Random Forest Baseline
    rf = BaselineModelTrainer("random_forest")
    rf.fit(X_train, y_train)
    rf_metrics = rf.evaluate(X_test, y_test)

    logger.info(f"HistGradientBoosting PR-AUC: {hgb_metrics['pr_auc']:.4f} | ROC-AUC: {hgb_metrics['roc_auc']:.4f}")
    logger.info(f"RandomForest PR-AUC: {rf_metrics['pr_auc']:.4f} | ROC-AUC: {rf_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

"""
Machine Learning Baseline Models (HistGradientBoosting, RandomForest).
Provides benchmark reference for deep learning evaluation.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from src.utils.logger import get_logger

logger = get_logger("BaselineModels")


class BaselineModelTrainer:
    """
    Trains and evaluates Scikit-Learn tabular baseline classifiers.
    """

    def __init__(self, model_type: str = "hist_gb"):
        self.model_type = model_type
        if model_type == "hist_gb":
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42,
                class_weight="balanced",
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported baseline model: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineModelTrainer":
        """Fit baseline classifier."""
        logger.info(f"Training baseline model ({self.model_type}) on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return spike probability (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        probs = self.predict_proba(X_test)
        preds = (probs >= 0.5).astype(int)

        try:
            roc_auc = float(roc_auc_score(y_test, probs))
        except Exception:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_test, probs))
        except Exception:
            pr_auc = 0.0

        f1 = float(f1_score(y_test, preds, zero_division=0))
        brier = float(brier_score_loss(y_test, probs))

        metrics = {
            "model": self.model_type,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "brier_score": brier,
        }
        logger.info(f"Baseline Results: {metrics}")
        return metrics

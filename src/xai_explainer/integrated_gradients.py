"""
Captum Integrated Gradients Explainer for Multi-Modal PyTorch Network.
Attributes predictions to specific time steps and NLP sentiment embedding features.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from src.models.hybrid_network import MarketPulseNet
from src.utils.logger import get_logger

logger = get_logger("IntegratedGradientsExplainer")


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients attribution for PyTorch multi-modal MarketPulseNet.
    """

    def __init__(self, model: MarketPulseNet, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def attribute(
        self,
        ts_input: torch.Tensor,     # [1, 78, 16]
        text_input: torch.Tensor,   # [1, 768]
        steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Compute path-integral gradient attributions from zero-baseline.
        """
        ts_input = ts_input.to(self.device).requires_grad_(True)
        text_input = text_input.to(self.device).requires_grad_(True)

        ts_baseline = torch.zeros_like(ts_input)
        text_baseline = torch.zeros_like(text_input)

        ts_grads = []
        text_grads = []

        for step in range(steps + 1):
            alpha = step / float(steps)
            ts_step = (ts_baseline + alpha * (ts_input - ts_baseline)).detach().requires_grad_(True)
            text_step = (text_baseline + alpha * (text_input - text_baseline)).detach().requires_grad_(True)

            logits, _ = self.model(ts_step, text_step)
            prob = torch.sigmoid(logits)

            prob.backward()

            if ts_step.grad is not None:
                ts_grads.append(ts_step.grad.detach().cpu().numpy())
            if text_step.grad is not None:
                text_grads.append(text_step.grad.detach().cpu().numpy())

        # Integrated gradients = (input - baseline) * average_gradients
        avg_ts_grad = np.mean(ts_grads, axis=0) if ts_grads else np.zeros_like(ts_input.detach().cpu().numpy())
        avg_text_grad = np.mean(text_grads, axis=0) if text_grads else np.zeros_like(text_input.detach().cpu().numpy())

        ts_attr = (ts_input.detach().cpu().numpy() - ts_baseline.detach().cpu().numpy()) * avg_ts_grad
        text_attr = (text_input.detach().cpu().numpy() - text_baseline.detach().cpu().numpy()) * avg_text_grad

        total_ts_score = float(np.sum(np.abs(ts_attr)))
        total_text_score = float(np.sum(np.abs(text_attr)))
        total_score = total_ts_score + total_text_score + 1e-9

        return {
            "time_series_attribution_pct": (total_ts_score / total_score) * 100.0,
            "sentiment_text_attribution_pct": (total_text_score / total_score) * 100.0,
            "ts_time_step_importance": np.sum(np.abs(ts_attr), axis=2).squeeze(0).tolist(),
        }

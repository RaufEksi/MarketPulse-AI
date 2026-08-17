"""
Captum Integrated Gradients Explainer for Multi-Modal PyTorch Network.
Attributes predictions to specific time steps and NLP sentiment embedding features.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

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
        ts_input: torch.Tensor,  # [1, 78, 16]
        text_input: torch.Tensor,  # [1, 768]
        steps: int = 20,
        feature_names: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute path-integral gradient attributions from zero-baseline.
        Also extracts Cross-Attention weights and per-feature importance.
        """
        ts_input = ts_input.to(self.device).requires_grad_(True)
        text_input = text_input.to(self.device).requires_grad_(True)

        # Extract Cross-Attention weights from forward pass
        with torch.no_grad():
            _, attn_weights = self.model(ts_input, text_input)
            # attn_weights shape: [Batch, 1, Seq_Len]
            attention_map = attn_weights.squeeze().cpu().numpy().tolist()

        ts_baseline = torch.zeros_like(ts_input)
        text_baseline = torch.zeros_like(text_input)

        ts_grads = []
        text_grads = []

        for step in range(steps + 1):
            alpha = step / float(steps)
            ts_step = (ts_baseline + alpha * (ts_input - ts_baseline)).detach().requires_grad_(True)
            text_step = (
                (text_baseline + alpha * (text_input - text_baseline)).detach().requires_grad_(True)
            )

            logits, _ = self.model(ts_step, text_step)
            prob = torch.sigmoid(logits)

            prob.backward()

            if ts_step.grad is not None:
                ts_grads.append(ts_step.grad.detach().cpu().numpy())
            if text_step.grad is not None:
                text_grads.append(text_step.grad.detach().cpu().numpy())

        # Integrated gradients = (input - baseline) * average_gradients
        avg_ts_grad = (
            np.mean(ts_grads, axis=0)
            if ts_grads
            else np.zeros_like(ts_input.detach().cpu().numpy())
        )
        avg_text_grad = (
            np.mean(text_grads, axis=0)
            if text_grads
            else np.zeros_like(text_input.detach().cpu().numpy())
        )

        ts_attr = (
            ts_input.detach().cpu().numpy() - ts_baseline.detach().cpu().numpy()
        ) * avg_ts_grad
        text_attr = (
            text_input.detach().cpu().numpy() - text_baseline.detach().cpu().numpy()
        ) * avg_text_grad

        total_ts_score = float(np.sum(np.abs(ts_attr)))
        total_text_score = float(np.sum(np.abs(text_attr)))
        total_score = total_ts_score + total_text_score + 1e-9

        # Sum attributions across sequence timesteps for each technical feature [16]
        raw_feature_attr = np.sum(np.abs(ts_attr), axis=1).squeeze(0)  # [16]
        feat_total = float(np.sum(raw_feature_attr)) + 1e-9

        feature_importance_dict = {}
        if feature_names and len(feature_names) == len(raw_feature_attr):
            for name, val in zip(feature_names, raw_feature_attr):
                feature_importance_dict[name] = float(val / feat_total)
        else:
            for idx, val in enumerate(raw_feature_attr):
                feature_importance_dict[f"feature_{idx}"] = float(val / feat_total)

        return {
            "time_series_attribution_pct": round((total_ts_score / total_score) * 100.0, 2),
            "sentiment_text_attribution_pct": round((total_text_score / total_score) * 100.0, 2),
            "ts_time_step_importance": np.sum(np.abs(ts_attr), axis=2).squeeze(0).tolist(),
            "attention_weights": attention_map,
            "feature_attributions": feature_importance_dict,
        }

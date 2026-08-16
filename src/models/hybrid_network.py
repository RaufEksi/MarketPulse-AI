"""
End-to-End MarketPulseNet Hybrid Deep Learning Network.
Combines TimeSeriesEncoder, TextProjectionEncoder, CrossAttentionFusion and a Classification Head.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from src.config.settings import get_settings
from src.models.time_series_branch import TimeSeriesEncoder
from src.models.text_branch import TextProjectionEncoder
from src.models.cross_attention import CrossAttentionFusion


class MarketPulseNet(nn.Module):
    """
    Multi-Modal Deep Neural Network for Short-Term Volatility Spike Prediction.
    """

    def __init__(
        self,
        ts_input_dim: int = 16,
        text_input_dim: int = 768,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 1,
        dropout: float = 0.3,
        ts_model_type: str = "bilstm",
    ):
        super().__init__()
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            model_type=ts_model_type,
        )
        self.text_encoder = TextProjectionEncoder(
            input_dim=text_input_dim,
            projected_dim=hidden_dim,
            dropout=dropout,
        )
        self.fusion = CrossAttentionFusion(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        ts_input: torch.Tensor,
        text_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ts_input: [Batch, Seq_Len=78, Num_Features=16]
            text_input: [Batch, 768]

        Returns:
            logits: [Batch, 1]
            attention_weights: [Batch, 1, Seq_Len]
        """
        ts_feats = self.ts_encoder(ts_input)  # [Batch, 78, 128]
        text_feats = self.text_encoder(text_input)  # [Batch, 1, 128]

        fused_rep, attn_weights = self.fusion(text_feats, ts_feats)  # [Batch, 128]
        logits = self.classifier(fused_rep)  # [Batch, 1]

        return logits.squeeze(-1), attn_weights

    def predict_probability(
        self, ts_input: torch.Tensor, text_input: torch.Tensor
    ) -> torch.Tensor:
        """Return sigmoid volatility spike probability (0.0 to 1.0)."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(ts_input, text_input)
            probs = torch.sigmoid(logits)
        return probs

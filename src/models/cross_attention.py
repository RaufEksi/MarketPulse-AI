"""
Multi-Head Cross-Attention Fusion Layer.
Allows text sentiment embeddings (Query) to attend over sequential price features (Key, Value).
"""

from typing import Tuple

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion module.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_emb: torch.Tensor,       # Query: [Batch, 1, Embed_Dim]
        ts_features: torch.Tensor,    # Key/Value: [Batch, Seq_Len, Embed_Dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_representation: [Batch, Embed_Dim]
            attention_weights: [Batch, 1, Seq_Len]
        """
        attn_out, attn_weights = self.cross_attn(
            query=text_emb,
            key=ts_features,
            value=ts_features,
            need_weights=True,
            average_attn_weights=True,
        )
        # Residual + Norm
        x = self.layer_norm1(text_emb + attn_out)

        # Feed Forward + Residual + Norm
        ff_out = self.feed_forward(x)
        out = self.layer_norm2(x + ff_out)

        # Squeeze sequence dimension
        return out.squeeze(1), attn_weights

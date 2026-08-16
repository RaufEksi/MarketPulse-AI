"""
PyTorch NLP Text Branch: FinBERT Dense Projection Encoder.
Maps 768-dimensional sentence embeddings to multi-modal latent dimension (e.g., 128).
"""

import torch
import torch.nn as nn


class TextProjectionEncoder(nn.Module):
    """
    Projects FinBERT embeddings into the common cross-modal feature dimension.
    """

    def __init__(
        self,
        input_dim: int = 768,
        projected_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, 768] or [Batch, Seq, 768]
        Output: [Batch, 1, Projected_Dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [Batch, 1, 768]
        return self.net(x)  # [Batch, 1, Projected_Dim]

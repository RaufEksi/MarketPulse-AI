"""
Multi-Modal Aligned Dataset & Walk-Forward DataLoader builder.
Builds PyTorch tensors: Time-Series [Batch, 78, 16], Text Embeddings [Batch, 768], Target [Batch].
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("DatasetBuilder")


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset returning aligned (price_sequence, text_embedding, target_label) tuples.
    """

    def __init__(
        self,
        time_series_sequences: np.ndarray,  # shape [N, seq_len, num_features]
        text_embeddings: np.ndarray,         # shape [N, text_dim]
        targets: np.ndarray,                 # shape [N]
    ):
        self.ts_data = torch.tensor(time_series_sequences, dtype=torch.float32)
        self.text_data = torch.tensor(text_embeddings, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ts_data[idx], self.text_data[idx], self.targets[idx]


def build_sliding_windows(
    feature_matrix: np.ndarray,
    aligned_text_matrix: np.ndarray,
    target_vector: np.ndarray,
    sequence_length: int = 78,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding sequence windows of length `sequence_length`.
    """
    num_samples = len(feature_matrix) - sequence_length
    if num_samples <= 0:
        raise ValueError(
            f"Not enough data rows ({len(feature_matrix)}) for sequence length ({sequence_length})"
        )

    num_features = feature_matrix.shape[1]
    text_dim = aligned_text_matrix.shape[1]

    ts_windows = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    text_windows = np.zeros((num_samples, text_dim), dtype=np.float32)
    y_windows = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        ts_windows[i] = feature_matrix[i : i + sequence_length]
        # Text embedding at current bar t (end of window)
        text_windows[i] = aligned_text_matrix[i + sequence_length - 1]
        y_windows[i] = target_vector[i + sequence_length - 1]

    # Filter out NaNs in targets
    valid_mask = ~np.isnan(y_windows)
    return ts_windows[valid_mask], text_windows[valid_mask], y_windows[valid_mask]


def create_walk_forward_dataloaders(
    ts_windows: np.ndarray,
    text_windows: np.ndarray,
    targets: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data chronologically (train -> validation -> test) to prevent lookahead leak.
    """
    n = len(targets)
    train_end = int(n * (1.0 - val_split - test_split))
    val_end = int(n * (1.0 - test_split))

    train_ds = MultiModalDataset(
        ts_windows[:train_end], text_windows[:train_end], targets[:train_end]
    )
    val_ds = MultiModalDataset(
        ts_windows[train_end:val_end],
        text_windows[train_end:val_end],
        targets[train_end:val_end],
    )
    test_ds = MultiModalDataset(
        ts_windows[val_end:], text_windows[val_end:], targets[val_end:]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        f"DataLoaders created - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader

"""
Exponential Decay Forward-Fill Temporal Alignment Engine.
Fuses irregular discrete NLP sentiment events into regular 5-minute price bars.
Formula: S_fused(t_bar) = sum_{i: t_i <= t_bar} [ S_i * exp(-lambda * (t_bar - t_i)) ]
"""

from typing import Optional
import numpy as np
import pandas as pd
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("TemporalAligner")


class TemporalAligner:
    """
    Implements vectorized exponential decay temporal alignment between text embeddings and OHLCV bars.
    """

    def __init__(self, decay_lambda_per_hour: Optional[float] = None):
        settings = get_settings()
        self.decay_lambda_per_hour = (
            decay_lambda_per_hour or settings.data.decay_lambda_per_hour
        )
        # Convert lambda to per-second decay rate: lambda_sec = lambda_hour / 3600
        self.decay_lambda_sec = self.decay_lambda_per_hour / 3600.0

    def align_sentiment_to_bars(
        self,
        bars_df: pd.DataFrame,
        text_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_dim: int = 768,
    ) -> np.ndarray:
        """
        Produce an aligned sentiment matrix of shape [num_bars, embedding_dim].

        Args:
            bars_df: DataFrame of 5-min bars with 'timestamp' column (UTC).
            text_df: DataFrame of text events with 'timestamp' column (UTC).
            embeddings: [num_texts, embedding_dim] FinBERT embedding matrix.
            embedding_dim: Embedding dimension (768).

        Returns:
            aligned_embeddings: [num_bars, embedding_dim] numpy array.
        """
        num_bars = len(bars_df)
        aligned_matrix = np.zeros((num_bars, embedding_dim), dtype=np.float32)

        if len(text_df) == 0 or len(embeddings) == 0:
            logger.info("No text events provided; returning zero-sentiment matrix.")
            return aligned_matrix

        # Ensure timestamps are in nanoseconds / seconds since epoch
        bar_timestamps = pd.to_datetime(bars_df["timestamp"]).astype(np.int64) // 10**9
        text_timestamps = pd.to_datetime(text_df["timestamp"]).astype(np.int64) // 10**9

        bar_ts_arr = bar_timestamps.values
        text_ts_arr = text_timestamps.values

        # For each bar, compute vectorized decaying sum of all previous events
        # To keep computation fast, only look back up to 24 hours (86400 seconds)
        max_lookback_sec = 86400

        for b_idx, t_bar in enumerate(bar_ts_arr):
            valid_mask = (text_ts_arr <= t_bar) & (text_ts_arr >= t_bar - max_lookback_sec)
            if not np.any(valid_mask):
                continue

            dt = t_bar - text_ts_arr[valid_mask]  # elapsed seconds
            weights = np.exp(-self.decay_lambda_sec * dt)[:, np.newaxis]  # shape [M, 1]

            selected_embeddings = embeddings[valid_mask]  # shape [M, 768]
            weighted_sum = np.sum(selected_embeddings * weights, axis=0)  # shape [768]
            weight_total = np.sum(weights) + 1e-9

            # Normalized weighted average
            aligned_matrix[b_idx] = weighted_sum / weight_total

        logger.info(f"Aligned {len(text_df)} text events into {num_bars} price bars.")
        return aligned_matrix

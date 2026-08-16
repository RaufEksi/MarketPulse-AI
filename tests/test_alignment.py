"""
Unit tests for Temporal Alignment and Dataset Building.
"""

import numpy as np
import pandas as pd
from src.data_alignment.exponential_decay import TemporalAligner
from src.data_alignment.dataset_builder import build_sliding_windows, create_walk_forward_dataloaders


def test_exponential_decay_aligner(sample_ohlcv_df: pd.DataFrame, sample_text_df: pd.DataFrame):
    aligner = TemporalAligner(decay_lambda_per_hour=0.5)
    mock_embeddings = np.random.normal(0, 1, size=(len(sample_text_df), 768)).astype(np.float32)

    aligned = aligner.align_sentiment_to_bars(
        bars_df=sample_ohlcv_df,
        text_df=sample_text_df,
        embeddings=mock_embeddings,
        embedding_dim=768,
    )

    assert aligned.shape == (len(sample_ohlcv_df), 768)
    assert not np.isnan(aligned).any()


def test_sliding_window_builder():
    n = 100
    features = np.random.normal(0, 1, size=(n, 16)).astype(np.float32)
    text_emb = np.random.normal(0, 1, size=(n, 768)).astype(np.float32)
    targets = np.random.choice([0.0, 1.0], size=n)

    ts_win, text_win, y_win = build_sliding_windows(features, text_emb, targets, sequence_length=78)

    assert ts_win.shape == (n - 78, 78, 16)
    assert text_win.shape == (n - 78, 768)
    assert y_win.shape == (n - 78,)

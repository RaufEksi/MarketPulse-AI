"""
Unit tests for Temporal Alignment and Dataset Building.
"""

import numpy as np
import pandas as pd

from src.data_alignment.dataset_builder import (
    build_sliding_windows,
    create_walk_forward_dataloaders,
)
from src.data_alignment.exponential_decay import TemporalAligner


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


def test_walk_forward_dataloaders():
    n = 50
    ts_win = np.random.normal(0, 1, size=(n, 78, 16)).astype(np.float32)
    text_win = np.random.normal(0, 1, size=(n, 768)).astype(np.float32)
    y_win = np.random.choice([0.0, 1.0], size=n).astype(np.float32)

    train_l, val_l, test_l = create_walk_forward_dataloaders(
        ts_win, text_win, y_win, val_split=0.2, test_split=0.2, batch_size=8
    )
    assert len(train_l.dataset) == 30
    assert len(val_l.dataset) == 10
    assert len(test_l.dataset) == 10


def test_aligner_decay_uses_real_seconds_for_any_datetime_unit():
    """Decay weights depend on elapsed seconds, whatever datetime64 resolution pandas picks."""
    # Arrange: one text 1h and one 2h before the bar, microsecond-resolution timestamps
    bar_time = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    bars = pd.DataFrame({"timestamp": pd.Series([bar_time]).astype("datetime64[us, UTC]")})
    texts = pd.DataFrame(
        {"timestamp": [bar_time - pd.Timedelta(hours=1), bar_time - pd.Timedelta(hours=2)]}
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Act
    aligned = TemporalAligner(decay_lambda_per_hour=1.0).align_sentiment_to_bars(
        bars, texts, embeddings, embedding_dim=2
    )

    # Assert: weights exp(-1) and exp(-2), normalized
    w1, w2 = np.exp(-1.0), np.exp(-2.0)
    np.testing.assert_allclose(aligned[0], [w1 / (w1 + w2), w2 / (w1 + w2)], rtol=1e-5)

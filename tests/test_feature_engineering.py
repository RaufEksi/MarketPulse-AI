"""
Unit tests for Technical Indicators and Feature Engineering.
"""

import pandas as pd

from src.feature_engineering.labeler import VolatilityLabeler
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine
from src.feature_engineering.text_preprocessor import TextPreprocessor


def test_technical_indicator_engine(sample_ohlcv_df: pd.DataFrame):
    engine = TechnicalFeatureEngine()
    transformed = engine.transform(sample_ohlcv_df)

    assert "atr_14" in transformed.columns
    assert "rsi_14" in transformed.columns
    assert "macd_hist" in transformed.columns
    assert "bb_pct_b" in transformed.columns
    assert "rolling_vol_12" in transformed.columns

    # No NaN in transformed indicators
    assert not transformed["atr_14"].isna().any()
    assert not transformed["rsi_14"].isna().any()


def test_text_preprocessor():
    preprocessor = TextPreprocessor()
    raw = "Breaking: $AAPL hits record high! Visit https://example.com/aapl for details <p>Now</p>"
    cleaned = preprocessor.clean(raw)

    assert "https" not in cleaned
    assert "<p>" not in cleaned
    assert "AAPL" in cleaned


def test_volatility_labeler(sample_ohlcv_df: pd.DataFrame):
    engine = TechnicalFeatureEngine()
    features = engine.transform(sample_ohlcv_df)

    labeler = VolatilityLabeler(horizon_bars=6, threshold_pct=0.15)
    labeled = labeler.create_labels(features)

    assert "atr_spike_target" in labeled.columns
    valid_labels = labeled["atr_spike_target"].dropna()
    assert set(valid_labels.unique()).issubset({0.0, 1.0})


def test_fallback_embeddings_are_stable_across_processes():
    """Hash-fallback embeddings must not depend on PYTHONHASHSEED (train vs. serve)."""
    import os
    import subprocess  # nosec B404
    import sys

    from src.feature_engineering.sentiment_embedder import FinBERTEmbedder

    code = (
        "from src.feature_engineering.sentiment_embedder import FinBERTEmbedder;"
        "print(FinBERTEmbedder()._fallback_embed(['Fed hikes rates'])[0, :3].tolist())"
    )
    outputs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.add(result.stdout.strip().splitlines()[-1])

    local = FinBERTEmbedder()._fallback_embed(["Fed hikes rates"])[0, :3].tolist()
    assert len(outputs) == 1
    assert outputs.pop() == str(local)

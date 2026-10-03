"""
Process-wide model and embedder cache shared by the /predict and /explain routes.

Loads the trained MarketPulseNet checkpoint from ``settings.model.checkpoint_path``.
When no checkpoint exists yet, falls back to an untrained network so the API still
boots, and logs a warning telling the operator to run ``scripts/train_model.py``.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data_alignment.exponential_decay import TemporalAligner
from src.feature_engineering.sentiment_embedder import FinBERTEmbedder
from src.feature_engineering.technical_indicators import (
    MODEL_FEATURE_COLUMNS,
    TechnicalFeatureEngine,
)
from src.models.checkpoint import ModelBundle, load_checkpoint
from src.models.hybrid_network import MarketPulseNet
from src.utils.logger import get_logger

logger = get_logger("ModelRegistry")

UNTRAINED_MODEL_SOURCE = "untrained"

_bundle: Optional[ModelBundle] = None
_embedder: Optional[FinBERTEmbedder] = None
_feature_engine = TechnicalFeatureEngine()
_aligner = TemporalAligner()


def _untrained_bundle() -> ModelBundle:
    """Random-weight model with identity normalization, used when no checkpoint exists."""
    settings = get_settings()
    model = MarketPulseNet(
        ts_input_dim=len(MODEL_FEATURE_COLUMNS),
        text_input_dim=settings.nlp.embedding_dim,
        hidden_dim=settings.model.time_series.hidden_dim,
    )
    model.eval()
    num_features = len(MODEL_FEATURE_COLUMNS)
    return ModelBundle(
        model=model,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        feature_mean=np.zeros(num_features, dtype=np.float32),
        feature_std=np.ones(num_features, dtype=np.float32),
        metadata={"source": UNTRAINED_MODEL_SOURCE},
    )


def get_model_bundle() -> ModelBundle:
    """
    Return the cached model bundle, loading the trained checkpoint on first use.

    Returns:
        ModelBundle for the trained checkpoint, or an untrained one if none exists.

    Raises:
        ModelInferenceError: If a checkpoint file exists but cannot be loaded.
    """
    global _bundle
    if _bundle is None:
        ckpt_path = Path(get_settings().model.checkpoint_path)
        if ckpt_path.is_file():
            _bundle = load_checkpoint(ckpt_path)
            _bundle.metadata.setdefault("source", str(ckpt_path))
        else:
            logger.warning(
                f"No trained checkpoint at {ckpt_path}; serving an UNTRAINED model. "
                "Run scripts/train_model.py to create one."
            )
            _bundle = _untrained_bundle()
    return _bundle


def get_embedder() -> FinBERTEmbedder:
    """
    Return the cached FinBERT embedder, warning if it differs from the training backend.

    Returns:
        Shared FinBERTEmbedder instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = FinBERTEmbedder()
        trained_with = get_model_bundle().metadata.get("text_embedder")
        if trained_with and trained_with != _embedder.backend:
            logger.warning(
                f"Model was trained with '{trained_with}' text embeddings but the API is "
                f"using '{_embedder.backend}'; sentiment inputs will not match training."
            )
    return _embedder


def reset_registry() -> None:
    """Drop cached model and embedder so the next request reloads them (used by tests)."""
    global _bundle, _embedder
    _bundle = None
    _embedder = None


def build_price_sequence(bars_df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    """
    Turn raw OHLCV bars into a normalized model input sequence.

    Args:
        bars_df: OHLCV bars with at least a 'close' column, oldest first.
        bundle: Model bundle providing feature order and normalization stats.

    Returns:
        Float32 array [lookback_bars, num_features], front-padded with the earliest bar.
    """
    lookback = get_settings().data.lookback_bars
    features_df = _feature_engine.transform(bars_df)
    for col in bundle.feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0

    raw = features_df[bundle.feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    if len(raw) < lookback:
        padding = np.repeat(raw[:1], lookback - len(raw), axis=0)
        raw = np.vstack([padding, raw])
    return bundle.normalize(raw[-lookback:])


def build_text_vector(
    bars_df: pd.DataFrame,
    texts: List[str],
    timestamps: List[object],
) -> np.ndarray:
    """
    Embed recent texts and decay-align them to the last bar.

    Args:
        bars_df: OHLCV bars with a 'timestamp' column; only the last bar is used.
        texts: Text strings (headlines, posts).
        timestamps: Publication time of each text.

    Returns:
        Float32 array [1, embedding_dim]; zeros when there is no text.
    """
    embedding_dim = get_settings().nlp.embedding_dim
    if not texts:
        return np.zeros((1, embedding_dim), dtype=np.float32)
    embeddings = get_embedder().embed_texts(texts)
    text_df = pd.DataFrame({"timestamp": timestamps, "text": texts})
    return _aligner.align_sentiment_to_bars(
        bars_df.tail(1), text_df, embeddings, embedding_dim=embedding_dim
    )

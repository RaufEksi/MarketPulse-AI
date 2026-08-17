"""
FastAPI /predict endpoint: Real-time volatility spike prediction.
"""

import time
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from src.api.schemas import PredictRequest, PredictResponse, ConfidenceInterval
from src.models.hybrid_network import MarketPulseNet
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine
from src.feature_engineering.sentiment_embedder import FinBERTEmbedder
from src.data_alignment.exponential_decay import TemporalAligner
from src.utils.logger import get_logger

logger = get_logger("PredictRoute")
router = APIRouter()

# Shared lazy model instance
_model = None
_embedder = None
_feature_engine = TechnicalFeatureEngine()
_aligner = TemporalAligner()


def get_or_load_model() -> MarketPulseNet:
    global _model
    if _model is None:
        _model = MarketPulseNet(ts_input_dim=16, text_input_dim=768, hidden_dim=128)
        _model.eval()
    return _model


def get_or_load_embedder() -> FinBERTEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FinBERTEmbedder()
    return _embedder


@router.post("/predict", response_model=PredictResponse)
async def predict_volatility(request: PredictRequest) -> PredictResponse:
    """
    Predict probability of a short-term volatility spike (>=15% ATR increase over next 30 minutes).
    """
    start_time = time.perf_counter()
    prediction_id = f"mp-{uuid.uuid4().hex[:12]}"

    if len(request.ohlcv_bars) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"At least 20 OHLCV bars required for feature extraction; received {len(request.ohlcv_bars)}",
        )

    # 1. Convert bars to DataFrame and compute technical indicators
    bars_data = [b.model_dump() for b in request.ohlcv_bars]
    bars_df = pd.DataFrame(bars_data)
    features_df = _feature_engine.transform(bars_df)

    # 2. Extract technical features array of shape [1, 78, 16]
    feature_cols = [
        "open", "high", "low", "close", "volume_ratio", "log_return",
        "atr_14", "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "rolling_vol_12", "rolling_vol_36", "rolling_vol_78"
    ]
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    raw_features = features_df[feature_cols].values
    if len(raw_features) < 78:
        # Pad sequence with earliest bar if shorter than 78
        padding = np.repeat(raw_features[:1], 78 - len(raw_features), axis=0)
        ts_seq = np.vstack([padding, raw_features])[-78:]
    else:
        ts_seq = raw_features[-78:]

    ts_tensor = torch.tensor(ts_seq, dtype=torch.float32).unsqueeze(0)  # [1, 78, 16]

    # 3. Process text events
    embedder = get_or_load_embedder()
    if request.recent_texts:
        text_strings = [t.headline for t in request.recent_texts]
        text_timestamps = [t.timestamp for t in request.recent_texts]
        text_df = pd.DataFrame({"timestamp": text_timestamps, "text": text_strings})
        embeddings = embedder.embed_texts(text_strings)
        aligned_sentiment = _aligner.align_sentiment_to_bars(bars_df.tail(1), text_df, embeddings)
        text_tensor = torch.tensor(aligned_sentiment, dtype=torch.float32)  # [1, 768]
    else:
        text_tensor = torch.zeros((1, 768), dtype=torch.float32)

    # 4. Forward pass
    model = get_or_load_model()
    with torch.no_grad():
        prob = float(model.predict_probability(ts_tensor, text_tensor).item())

    # 5. Risk classification
    if prob >= 0.70:
        risk_level = "CRITICAL_VOLATILITY"
    elif prob >= 0.40:
        risk_level = "MODERATE_VOLATILITY"
    else:
        risk_level = "LOW_VOLATILITY"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictResponse(
        prediction_id=prediction_id,
        symbol=request.symbol,
        timestamp=request.ohlcv_bars[-1].timestamp,
        volatility_spike_probability=round(prob, 4),
        risk_level=risk_level,
        confidence_interval=ConfidenceInterval(
            lower=round(max(0.0, prob - 0.05), 4),
            upper=round(min(1.0, prob + 0.05), 4),
        ),
        inference_latency_ms=round(latency_ms, 2),
    )

"""
FastAPI /predict endpoint: Real-time volatility spike prediction.
"""

import time
import uuid

import pandas as pd
import torch
from fastapi import APIRouter, HTTPException

from src.api.model_registry import build_price_sequence, build_text_vector, get_model_bundle
from src.api.schemas import ConfidenceInterval, PredictRequest, PredictResponse
from src.utils.exceptions import ModelInferenceError
from src.utils.logger import get_logger

logger = get_logger("PredictRoute")
router = APIRouter()


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
            detail=(
                "At least 20 OHLCV bars required for feature extraction; "
                f"received {len(request.ohlcv_bars)}"
            ),
        )

    # 1. Technical features -> normalized [1, lookback, num_features] sequence
    try:
        bundle = get_model_bundle()
    except ModelInferenceError as e:
        logger.error(f"Model unavailable: {e.message}")
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.") from e
    bars_df = pd.DataFrame([b.model_dump() for b in request.ohlcv_bars])
    ts_seq = build_price_sequence(bars_df, bundle)
    ts_tensor = torch.tensor(ts_seq, dtype=torch.float32).unsqueeze(0)

    # 2. Text events -> FinBERT embeddings decay-aligned to the last bar [1, 768]
    text_vec = build_text_vector(
        bars_df,
        [t.headline for t in request.recent_texts],
        [t.timestamp for t in request.recent_texts],
    )
    text_tensor = torch.tensor(text_vec, dtype=torch.float32)

    # 3. Forward pass
    with torch.no_grad():
        prob = float(bundle.model.predict_probability(ts_tensor, text_tensor).item())

    # 4. Risk classification
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

"""
FastAPI /backtest endpoint: Quantitative strategy backtesting simulation.

Loads recent 5-minute bars for the requested symbol, scores every bar with
MarketPulseNet over a rolling lookback window, and simulates the volatility
hedging strategy on those model probabilities. Without Alpaca credentials the
connector falls back to synthetic bars, which the response flags via ``is_synthetic``.
"""

from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException

from src.api.model_registry import get_model_bundle
from src.api.schemas import BacktestRequest, BacktestResponse
from src.config.settings import get_settings
from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine
from src.models.backtester import VolatilityBacktester
from src.models.checkpoint import ModelBundle
from src.models.hybrid_network import MarketPulseNet
from src.utils.exceptions import DataIngestionError, ModelInferenceError
from src.utils.logger import get_logger

logger = get_logger("BacktestRoute")
router = APIRouter()

# Same window as GET /market/bars: spans weekends/holidays and still yields several sessions
BARS_LOOKBACK_DAYS = 5
ALPACA_MAX_LIMIT = 10000
INFERENCE_BATCH_SIZE = 256

_feature_engine = TechnicalFeatureEngine()


def get_backtest_model() -> ModelBundle:
    """
    Return the model bundle used to score backtest bars.

    Shares the /predict and /explain registry, so backtests use the trained checkpoint
    (and its feature normalization) whenever one exists.

    Returns:
        ModelBundle with an eval-mode model, feature order and normalization stats.

    Raises:
        ModelInferenceError: If a checkpoint exists but cannot be loaded.
    """
    return get_model_bundle()


def _load_bars(symbol: str) -> Tuple[pd.DataFrame, bool]:
    """Fetch recent 5-min bars for a symbol; returns (bars sorted by time, is_synthetic)."""
    collector = AlpacaDataCollector()
    is_synthetic = not (collector.api_key and collector.secret_key)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=BARS_LOOKBACK_DAYS)
    df = collector.fetch_5min_bars(symbol, start_time, end_time, limit=ALPACA_MAX_LIMIT)
    return df.sort_values("timestamp").reset_index(drop=True), is_synthetic


def build_feature_windows(
    features_df: pd.DataFrame, bundle: ModelBundle, seq_len: int
) -> np.ndarray:
    """
    Slice a feature DataFrame into overlapping, normalized model input windows.

    Window ``i`` ends at bar ``i + seq_len - 1`` and uses only bars up to and
    including it, so a prediction never sees future data.

    Args:
        features_df: Output of ``TechnicalFeatureEngine.transform`` [N rows].
        bundle: Model bundle providing feature order and normalization stats.
        seq_len: Number of bars per window.

    Returns:
        Float32 array of shape [N - seq_len + 1, seq_len, num_features].

    Raises:
        ValueError: If the DataFrame has fewer than ``seq_len`` rows.
    """
    if len(features_df) < seq_len:
        raise ValueError(f"Need at least {seq_len} bars, got {len(features_df)}")

    frame = features_df.reindex(columns=bundle.feature_columns, fill_value=0.0).fillna(0.0)
    values = bundle.normalize(frame.to_numpy(dtype=np.float64))
    # sliding_window_view puts the window axis last: [N - seq_len + 1, features, seq_len]
    windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
    return np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)


def predict_window_probabilities(
    model: MarketPulseNet,
    windows: np.ndarray,
    text_dim: int,
) -> np.ndarray:
    """
    Score each window with the model, without sentiment input.

    Args:
        model: Model exposing ``predict_probability(ts_input, text_input)``.
        windows: Array [num_windows, seq_len, num_features].
        text_dim: Size of the text embedding the model expects.

    Returns:
        Spike probabilities [num_windows].
    """
    probs = []
    for start in range(0, len(windows), INFERENCE_BATCH_SIZE):
        ts_batch = torch.from_numpy(windows[start : start + INFERENCE_BATCH_SIZE])
        text_batch = torch.zeros((len(ts_batch), text_dim), dtype=torch.float32)
        with torch.no_grad():
            batch_probs = model.predict_probability(ts_batch, text_batch)
        probs.append(batch_probs.reshape(-1).cpu().numpy())
    return np.concatenate(probs).astype(np.float64)


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Simulate a risk-avoidance hedging strategy on the model's volatility predictions.

    Declared sync so FastAPI runs the data fetch and inference in its threadpool.

    Args:
        request: Symbol and strategy parameters.

    Returns:
        Strategy and buy-and-hold metrics and equity curves for the symbol.

    Raises:
        HTTPException: 502 if the data connector fails, 422 if too few bars are
            available, 503 if the model cannot be loaded.
    """
    settings = get_settings()
    seq_len = settings.data.lookback_bars
    symbol = request.symbol.upper()

    try:
        bars_df, is_synthetic = _load_bars(symbol)
    except DataIngestionError as e:
        raise HTTPException(status_code=502, detail=e.message) from e

    # Need one full window plus at least one forward return to simulate
    if len(bars_df) < seq_len + 1:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {seq_len + 1} bars for {symbol}; got {len(bars_df)}",
        )

    try:
        bundle = get_backtest_model()
    except ModelInferenceError as e:
        logger.error(f"Model unavailable: {e.message}")
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.") from e

    features_df = _feature_engine.transform(bars_df)
    windows = build_feature_windows(features_df, bundle, seq_len)
    probs = predict_window_probabilities(bundle.model, windows, settings.nlp.embedding_dim)
    # Prices aligned with the bar each window ends on
    prices = features_df["close"].to_numpy(dtype=np.float64)[seq_len - 1 :]

    logger.info(
        "Backtest scored",
        extra={"symbol": symbol, "bars": len(prices), "is_synthetic": is_synthetic},
    )

    backtester = VolatilityBacktester(
        spike_threshold=request.spike_threshold,
        hedge_reduction_factor=request.hedge_reduction_factor,
    )
    result = backtester.run(
        close_prices=prices,
        predicted_probabilities=probs,
        initial_capital=request.initial_capital,
    )

    return BacktestResponse(
        symbol=symbol,
        is_synthetic=is_synthetic,
        strategy_metrics=result.get("strategy_metrics", {}),
        benchmark_metrics=result.get("benchmark_metrics", {}),
        strategy_equity=result.get("strategy_equity", []),
        benchmark_equity=result.get("benchmark_equity", []),
        predicted_probabilities=probs.round(4).tolist(),
    )

"""
FastAPI /market endpoints: recent OHLCV bars and sentiment texts for a symbol.

These endpoints let the Streamlit dashboard obtain model inputs over HTTP instead of
importing the data engine or feature engineering layers directly. Connectors fall back
to synthetic data when API keys are not configured.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import MarketBar, MarketBarsResponse, MarketText, MarketTextsResponse
from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.data_engine.news_collector import NewsCollector
from src.data_engine.reddit_collector import RedditCollector
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("MarketDataRoute")
router = APIRouter(prefix="/market")

# Look back far enough to span weekends/holidays and still return a full session of bars
BARS_LOOKBACK_DAYS = 5
ALPACA_MAX_LIMIT = 10000
MAX_BARS = 1000
MAX_TEXTS = 100

_feature_engine = TechnicalFeatureEngine()


@router.get("/bars", response_model=MarketBarsResponse)
async def get_market_bars(
    symbol: str = Query("SPY", min_length=1, max_length=10),
    limit: int = Query(78, ge=20, le=MAX_BARS),
) -> MarketBarsResponse:
    """
    Return the most recent 5-minute OHLCV bars for a symbol with ATR(14) and RSI(14).

    Args:
        symbol: Ticker symbol.
        limit: Number of most recent bars to return.

    Returns:
        MarketBarsResponse with bars ordered oldest to newest.

    Raises:
        HTTPException: 502 if the data connector fails, 404 if no bars are available.
    """
    symbol = symbol.upper()
    collector = AlpacaDataCollector()
    is_synthetic = not (collector.api_key and collector.secret_key)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=BARS_LOOKBACK_DAYS)

    try:
        df = collector.fetch_5min_bars(symbol, start_time, end_time, limit=ALPACA_MAX_LIMIT)
    except DataIngestionError as e:
        raise HTTPException(status_code=502, detail=e.message) from e

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No bars available for {symbol}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    features_df = _feature_engine.transform(df).tail(limit)

    bars = [
        MarketBar(
            timestamp=row["timestamp"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=float(row["vwap"]) if pd.notna(row.get("vwap")) else None,
            trade_count=int(row["trade_count"]) if pd.notna(row.get("trade_count")) else None,
            atr_14=float(row["atr_14"]),
            rsi_14=float(row["rsi_14"]),
        )
        for _, row in features_df.iterrows()
    ]
    return MarketBarsResponse(symbol=symbol, is_synthetic=is_synthetic, bars=bars)


@router.get("/texts", response_model=MarketTextsResponse)
async def get_market_texts(
    symbol: str = Query("SPY", min_length=1, max_length=10),
    limit: int = Query(20, ge=1, le=MAX_TEXTS),
) -> MarketTextsResponse:
    """
    Return recent news headlines and Reddit posts mentioning a symbol, newest first.

    A failing source is logged and skipped so one outage does not hide the other.

    Args:
        symbol: Ticker symbol.
        limit: Maximum number of texts to return.

    Returns:
        MarketTextsResponse with texts ordered newest to oldest.
    """
    symbol = symbol.upper()
    frames: List[pd.DataFrame] = []

    try:
        frames.append(NewsCollector().fetch_headlines([symbol], page_size=limit))
    except DataIngestionError as e:
        logger.warning("News source unavailable", extra={"symbol": symbol, "error": e.message})

    try:
        frames.append(RedditCollector().fetch_posts([symbol], limit=limit))
    except DataIngestionError as e:
        logger.warning("Reddit source unavailable", extra={"symbol": symbol, "error": e.message})

    frames = [f for f in frames if not f.empty]
    if not frames:
        return MarketTextsResponse(symbol=symbol, texts=[])

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp", ascending=False).head(limit)

    texts = [
        MarketText(
            timestamp=row["timestamp"],
            symbol=str(row["symbol"]),
            source=str(row["source"]),
            text=str(row["text"]),
            score=float(row["score"]) if pd.notna(row.get("score")) else None,
        )
        for _, row in df.iterrows()
    ]
    return MarketTextsResponse(symbol=symbol, texts=texts)

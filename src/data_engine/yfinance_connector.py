"""
Yahoo Finance 5-minute OHLCV fallback data collector.
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("YFinanceDataCollector")


class YFinanceDataCollector:
    """
    Fallback intraday data collector using Yahoo Finance.
    """

    def fetch_5min_bars(
        self,
        symbol: str,
        period: str = "5d",
    ) -> pd.DataFrame:
        """
        Fetch 5-minute bars via yfinance.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="5m")

            if df.empty:
                logger.warning(f"No yfinance data found for {symbol}")
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(
                columns={
                    "Datetime": "timestamp",
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["symbol"] = symbol
            df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            df["trade_count"] = 0
            return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]
        except Exception as e:
            logger.error(f"Failed to fetch yfinance bars for {symbol}: {str(e)}")
            raise DataIngestionError(f"yfinance error: {str(e)}")

"""
Alpaca Markets 5-minute OHLCV bar collector.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import requests
from src.config.settings import get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("AlpacaDataCollector")


class AlpacaDataCollector:
    """
    Ingests historical and real-time intraday OHLCV bar data from Alpaca Markets Data API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.data_base_url = "https://data.alpaca.markets/v2"

    def fetch_5min_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch 5-minute OHLCV bars for a specified symbol and time window.
        """
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca credentials missing; generating synthetic fallback data.")
            return self._generate_synthetic_bars(symbol, start_time, end_time or datetime.now(timezone.utc))

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        url = f"{self.data_base_url}/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        params = {
            "timeframe": "5Min",
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "limit": limit,
            "adjustment": "raw",
            "feed": "sip",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("bars", [])

            if not data:
                logger.warning(f"No bars returned for {symbol} between {start_time} and {end_time}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"])

            df = pd.DataFrame(data)
            df = df.rename(
                columns={
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "trade_count",
                }
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca bars for {symbol}: {str(e)}")
            raise DataIngestionError(f"Alpaca API error: {str(e)}")

    def _generate_synthetic_bars(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generates realistic synthetic 5-min bars for testing/offline simulation."""
        import numpy as np
        date_range = pd.date_range(start=start_time, end=end_time, freq="5min", tz="UTC")
        if len(date_range) == 0:
            date_range = pd.date_range(start=start_time, periods=78, freq="5min", tz="UTC")

        n = len(date_range)
        np.random.seed(42)
        base_price = 500.0 if symbol == "SPY" else 180.0
        returns = np.random.normal(0.0001, 0.002, size=n)
        prices = base_price * np.exp(np.cumsum(returns))

        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.001, size=n)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.001, size=n)))
        opens = (prices + lows) / 2.0
        closes = prices
        volumes = np.random.randint(10000, 500000, size=n)

        return pd.DataFrame({
            "timestamp": date_range,
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": (opens + highs + lows + closes) / 4.0,
            "trade_count": np.random.randint(100, 5000, size=n),
        })

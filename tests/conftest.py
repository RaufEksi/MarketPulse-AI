"""
Pytest configuration, shared fixtures and mock data generators.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Fixture providing 100 5-min OHLCV bars."""
    np.random.seed(42)
    n = 100
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(minutes=5 * (n - i)) for i in range(n)]

    base_price = 500.0
    returns = np.random.normal(0.0001, 0.002, size=n)
    prices = base_price * np.exp(np.cumsum(returns))

    highs = prices * 1.002
    lows = prices * 0.998
    opens = (prices + lows) / 2.0
    closes = prices
    volumes = np.random.randint(10000, 100000, size=n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": (opens + highs + lows + closes) / 4.0,
            "trade_count": np.random.randint(50, 500, size=n),
        }
    )


@pytest.fixture
def sample_text_df() -> pd.DataFrame:
    """Fixture providing sample financial text headlines."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "id": "t1",
                "timestamp": now - timedelta(minutes=45),
                "symbol": "SPY",
                "source": "news/Reuters",
                "text": "FOMC member hints at interest rate cuts ahead.",
                "score": 100,
            },
            {
                "id": "t2",
                "timestamp": now - timedelta(minutes=15),
                "symbol": "SPY",
                "source": "reddit/r/wallstreetbets",
                "text": "Call option buying surges on SPY 550 strikes.",
                "score": 250,
            },
        ]
    )

"""
Unit tests for data engine connectors and storage manager.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import pytest
from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.data_engine.yfinance_connector import YFinanceDataCollector
from src.data_engine.reddit_collector import RedditCollector
from src.data_engine.news_collector import NewsCollector
from src.data_engine.storage_manager import StorageManager


def test_alpaca_synthetic_fallback():
    collector = AlpacaDataCollector(api_key=None, secret_key=None)
    start_time = datetime.now(timezone.utc) - timedelta(days=2)
    end_time = datetime.now(timezone.utc)
    df = collector.fetch_5min_bars("SPY", start_time, end_time)

    assert not df.empty
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "vwap" in df.columns
    assert df["symbol"].iloc[0] == "SPY"


def test_reddit_collector_synthetic():
    collector = RedditCollector(client_id=None, client_secret=None)
    df = collector.fetch_posts(["SPY", "QQQ"], limit=10)

    assert not df.empty
    assert "symbol" in df.columns
    assert "text" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 10


def test_news_collector_synthetic():
    collector = NewsCollector(api_key=None)
    df = collector.fetch_headlines(["SPY"], page_size=15)

    assert not df.empty
    assert "text" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 15


def test_storage_manager_lifecycle(tmp_path: Path):
    storage = StorageManager(base_dir=str(tmp_path))
    sample_df = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "open": [500.0],
        "high": [505.0],
        "low": [495.0],
        "close": [502.0],
        "volume": [10000],
        "vwap": [501.0],
        "trade_count": [100],
    })

    # Save and Load raw bars
    saved_path = storage.save_raw_bars(sample_df, "SPY")
    assert saved_path.exists()

    loaded_df = storage.load_raw_bars("SPY")
    assert len(loaded_df) == 1
    assert loaded_df["close"].iloc[0] == 502.0

    # Save and Load processed dataset
    proc_path = storage.save_processed_dataset(sample_df, "test_dataset")
    assert proc_path.exists()

    loaded_proc = storage.load_processed_dataset("test_dataset")
    assert len(loaded_proc) == 1

    # Manifest generation
    manifest_path = storage.save_manifest({"symbol": "SPY", "records": 100})
    assert manifest_path.exists()

    # Retention policy test
    cleaned = storage.clean_retention_policy(max_age_days=10)
    assert isinstance(cleaned, int)


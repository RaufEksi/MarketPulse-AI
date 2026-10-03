"""
Unit tests for data engine connectors and storage manager.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.data_engine.news_collector import NewsCollector
from src.data_engine.reddit_collector import RedditCollector
from src.data_engine.storage_manager import StorageManager
from src.utils.exceptions import DataIngestionError


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
    collector = RedditCollector(use_synthetic=True)
    df = collector.fetch_posts(["SPY", "QQQ"], limit=10)

    assert not df.empty
    assert "symbol" in df.columns
    assert "text" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 10


def _response(
    status: int, items: List[Dict[str, Any]], headers: Optional[Dict[str, str]] = None
) -> MagicMock:
    """Build a fake Arctic Shift HTTP response."""
    response = MagicMock(status_code=status, headers=headers or {})
    response.json.return_value = {"data": items, "error": None}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    return response


def test_reddit_collector_arctic_shift_posts_and_comments():
    """Posts and comments are mapped to the output schema; non-matching tickers are dropped."""
    # Arrange
    posts = [
        {
            "id": "p1",
            "created_utc": 1759500000,
            "title": "$NVDA to the moon",
            "selftext": "[removed]",
            "score": 120,
            "num_comments": 45,
        },
        {
            "id": "p2",
            "created_utc": 1759490000,
            "title": "NVDAX is not NVDA-free",
            "selftext": "",
            "score": 3,
            "num_comments": 1,
        },
        {
            "id": "p3",
            "created_utc": 1759480000,
            "title": "Thoughts on NVDAX?",
            "selftext": "",
            "score": 1,
            "num_comments": 0,
        },
    ]
    comments = [{"id": "c1", "created_utc": 1759495000, "body": "Bought NVDA puts", "score": 7}]
    session = MagicMock()
    session.get.side_effect = [_response(200, posts), _response(200, comments)]
    collector = RedditCollector(
        base_url="https://example.test", use_synthetic=False, session=session
    )

    # Act
    df = collector.fetch_posts(["NVDA"], subreddits=["wallstreetbets"], limit=10)

    # Assert
    assert list(df.columns) == [
        "id",
        "timestamp",
        "symbol",
        "source",
        "text",
        "score",
        "num_comments",
    ]
    assert set(df["id"]) == {"p1", "p2", "c1"}
    p1 = df.set_index("id").loc["p1"]
    assert p1["text"] == "$NVDA to the moon"
    assert p1["num_comments"] == 45
    c1 = df.set_index("id").loc["c1"]
    assert c1["source"] == "reddit/r/wallstreetbets/comments"
    assert c1["timestamp"] == datetime.fromtimestamp(1759495000, timezone.utc)
    first_params = session.get.call_args_list[0].kwargs["params"]
    assert first_params["subreddit"] == "wallstreetbets"
    assert first_params["query"] == "NVDA"


def test_reddit_collector_paginates_backwards_in_time():
    """A full page triggers another request with `before` set to the oldest item seen."""
    # Arrange
    page1 = [
        {"id": f"a{i}", "created_utc": 2000 - i, "title": "SPY", "score": 1, "num_comments": 0}
        for i in range(100)
    ]
    page2 = [{"id": "b0", "created_utc": 1800, "title": "SPY calls", "score": 1, "num_comments": 0}]
    session = MagicMock()
    session.get.side_effect = [_response(200, page1), _response(200, page2)]
    collector = RedditCollector(
        base_url="https://example.test", use_synthetic=False, session=session
    )
    after = datetime.fromtimestamp(1000, timezone.utc)

    # Act
    df = collector.fetch_posts(
        ["SPY"], subreddits=["stocks"], limit=150, after=after, include_comments=False
    )

    # Assert
    assert len(df) == 101
    second_params = session.get.call_args_list[1].kwargs["params"]
    assert second_params["before"] == 1901
    assert second_params["after"] == 1000
    assert second_params["limit"] == 50


def test_reddit_collector_retries_once_on_rate_limit():
    """A 429 waits for X-RateLimit-Reset and retries the same request."""
    # Arrange
    session = MagicMock()
    session.get.side_effect = [
        _response(429, [], {"X-RateLimit-Reset": "2"}),
        _response(200, []),
    ]
    collector = RedditCollector(
        base_url="https://example.test", use_synthetic=False, session=session
    )

    # Act
    with patch("src.data_engine.reddit_collector.time.sleep") as sleep:
        df = collector.fetch_posts(["SPY"], subreddits=["stocks"], include_comments=False)

    # Assert
    sleep.assert_called_once_with(2.0)
    assert df.empty
    assert session.get.call_count == 2


def test_reddit_collector_http_error_raises():
    """HTTP failures surface as DataIngestionError."""
    # Arrange
    session = MagicMock()
    session.get.return_value = _response(500, [])
    collector = RedditCollector(
        base_url="https://example.test", use_synthetic=False, session=session
    )

    # Act / Assert
    with pytest.raises(DataIngestionError):
        collector.fetch_posts(["SPY"], subreddits=["stocks"])


def test_news_collector_synthetic():
    collector = NewsCollector(api_key=None)
    df = collector.fetch_headlines(["SPY"], page_size=15)

    assert not df.empty
    assert "text" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 15


def test_storage_manager_lifecycle(tmp_path: Path):
    storage = StorageManager(base_dir=str(tmp_path))
    sample_df = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "open": [500.0],
            "high": [505.0],
            "low": [495.0],
            "close": [502.0],
            "volume": [10000],
            "vwap": [501.0],
            "trade_count": [100],
        }
    )

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

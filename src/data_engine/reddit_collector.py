"""
Reddit sentiment collector for financial subreddits, backed by the Arctic Shift archive.

Reddit closed self-service Data API app creation in November 2025, so PRAW credentials
are no longer obtainable for this project. This module instead queries the public
Arctic Shift HTTP API (https://github.com/ArthurHeitmann/arctic_shift), which mirrors
Reddit posts and comments and needs no API key. Both submissions and comments are
collected, since most of the retail sentiment signal lives in comment threads.
"""

import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.config.settings import get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("RedditCollector")

OUTPUT_COLUMNS = ["id", "timestamp", "symbol", "source", "text", "score", "num_comments"]
POSTS_ENDPOINT = "/api/posts/search"
COMMENTS_ENDPOINT = "/api/comments/search"
POST_FIELDS = "id,created_utc,title,selftext,score,num_comments"
COMMENT_FIELDS = "id,created_utc,body,score"
MAX_PAGE_SIZE = 100  # Arctic Shift caps `limit` at 100 per request
HTTP_TOO_MANY_REQUESTS = 429
MAX_RATE_LIMIT_WAIT_S = 60.0
REMOVED_MARKERS = {"[removed]", "[deleted]"}


class RedditCollector:
    """
    Collects financial sentiment text from targeted subreddits (r/wallstreetbets, r/stocks).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        use_synthetic: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize the collector.

        Args:
            base_url: Arctic Shift API base URL (defaults to settings).
            use_synthetic: If True, return synthetic posts instead of calling the API
                (defaults to settings).
            timeout_s: Per-request HTTP timeout in seconds (defaults to settings).
            session: Optional requests session, injectable for testing.
        """
        settings = get_settings()
        self.base_url = (base_url or settings.data.reddit_api_base_url).rstrip("/")
        self.use_synthetic = (
            settings.data.reddit_use_synthetic if use_synthetic is None else use_synthetic
        )
        self.timeout_s = timeout_s or settings.data.reddit_request_timeout_s
        self.default_subreddits = list(settings.data.reddit_subreddits)
        self.session = session or requests.Session()

    def fetch_posts(
        self,
        symbols: List[str],
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        include_comments: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch submissions (and optionally comments) mentioning the given ticker symbols.

        Args:
            symbols: Ticker symbols to search for (e.g. ["SPY", "NVDA"]).
            subreddits: Subreddits to search (defaults to settings).
            limit: Maximum items fetched per subreddit, symbol and kind (post/comment).
            after: Only return items created after this UTC time.
            before: Only return items created before this UTC time.
            include_comments: Whether to also collect matching comments.

        Returns:
            DataFrame with columns id, timestamp, symbol, source, text, score, num_comments.

        Raises:
            DataIngestionError: If the Arctic Shift API request fails.
        """
        if self.use_synthetic:
            logger.warning("Reddit synthetic mode enabled; generating synthetic sentiment text.")
            return self._generate_synthetic_posts(symbols, limit)

        records: List[Dict[str, Any]] = []
        for sub_name in subreddits or self.default_subreddits:
            for symbol in symbols:
                pattern = _ticker_pattern(symbol)
                posts = self._search(
                    POSTS_ENDPOINT,
                    {"subreddit": sub_name, "query": symbol, "fields": POST_FIELDS},
                    limit,
                    after,
                    before,
                )
                for post in posts:
                    text = _join_text(post.get("title"), post.get("selftext"))
                    if text and pattern.search(text):
                        records.append(
                            _record(post, symbol, f"reddit/r/{sub_name}", text, "num_comments")
                        )

                if not include_comments:
                    continue
                comments = self._search(
                    COMMENTS_ENDPOINT,
                    {"subreddit": sub_name, "body": symbol, "fields": COMMENT_FIELDS},
                    limit,
                    after,
                    before,
                )
                for comment in comments:
                    text = _join_text(comment.get("body"))
                    if text and pattern.search(text):
                        records.append(
                            _record(comment, symbol, f"reddit/r/{sub_name}/comments", text, None)
                        )

        if not records:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        df = df.drop_duplicates(subset=["id", "symbol"]).sort_values("timestamp")
        logger.info("Reddit items fetched", extra={"symbols": symbols, "count": len(df)})
        return df.reset_index(drop=True)

    def _search(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_items: int,
        after: Optional[datetime],
        before: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        """Page backwards in time through an Arctic Shift search endpoint."""
        items: List[Dict[str, Any]] = []
        cursor = before
        while len(items) < max_items:
            page_size = min(MAX_PAGE_SIZE, max_items - len(items))
            query = {**params, "limit": page_size, "sort": "desc"}
            if after is not None:
                query["after"] = int(after.timestamp())
            if cursor is not None:
                query["before"] = int(cursor.timestamp())

            page = self._get(endpoint, query)
            items.extend(page)
            if len(page) < page_size:
                break
            oldest = min(float(item["created_utc"]) for item in page)
            cursor = datetime.fromtimestamp(oldest, timezone.utc)
        return items

    def _get(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform one GET request, waiting once on a 429 rate-limit response."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_s)
            if response.status_code == HTTP_TOO_MANY_REQUESTS:
                wait_s = min(
                    float(response.headers.get("X-RateLimit-Reset", MAX_RATE_LIMIT_WAIT_S)),
                    MAX_RATE_LIMIT_WAIT_S,
                )
                logger.warning("Arctic Shift rate limited", extra={"wait_s": wait_s})
                time.sleep(wait_s)
                response = self.session.get(url, params=params, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Failed to query Arctic Shift: {str(e)}")
            raise DataIngestionError(f"Reddit (Arctic Shift) error: {str(e)}") from e

        if payload.get("error"):
            raise DataIngestionError(f"Reddit (Arctic Shift) error: {payload['error']}")
        data = payload.get("data") or []
        return list(data)

    def _generate_synthetic_posts(self, symbols: List[str], count: int) -> pd.DataFrame:
        """Generate realistic synthetic Reddit posts for testing."""
        headlines = [
            "Massive call option buying detected before earnings announcement!",
            "Why I am hedging my tech positions before tomorrow's CPI print.",
            "Rumors circulating regarding antitrust investigation on tech leaders.",
            "Record quarterly revenue beats expectations by wide margin.",
            "Is the sudden volatility spike a buying opportunity or a warning sign?",
        ]
        records = []
        now = datetime.now(timezone.utc)
        for i in range(count):
            symbol = random.choice(symbols)
            records.append(
                {
                    "id": f"reddit_syn_{i}",
                    "timestamp": now - timedelta(minutes=random.randint(1, 300)),
                    "symbol": symbol,
                    "source": "reddit/r/wallstreetbets",
                    "text": f"${symbol} - {random.choice(headlines)}",
                    "score": random.randint(5, 1200),
                    "num_comments": random.randint(2, 450),
                }
            )
        return pd.DataFrame(records)


def _ticker_pattern(symbol: str) -> "re.Pattern[str]":
    """Match a ticker as a standalone token, optionally prefixed with '$'."""
    return re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(symbol)}(?![A-Za-z0-9])")


def _join_text(*parts: Optional[str]) -> str:
    """Join non-empty text parts, dropping Reddit's removed/deleted placeholders."""
    kept = [p.strip() for p in parts if p and p.strip() and p.strip() not in REMOVED_MARKERS]
    return "\n".join(kept)


def _record(
    item: Dict[str, Any],
    symbol: str,
    source: str,
    text: str,
    comments_key: Optional[str],
) -> Dict[str, Any]:
    """Map an Arctic Shift item to the collector's output schema."""
    return {
        "id": item["id"],
        "timestamp": datetime.fromtimestamp(float(item["created_utc"]), timezone.utc),
        "symbol": symbol,
        "source": source,
        "text": text,
        "score": int(item.get("score") or 0),
        "num_comments": int(item.get(comments_key) or 0) if comments_key else 0,
    }

"""
Reddit PRAW sentiment and discussion scraper for financial subreddits.
"""

from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
from src.config.settings import get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("RedditCollector")


class RedditCollector:
    """
    Collects financial sentiment text from targeted subreddits (r/wallstreetbets, r/stocks).
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        settings = get_settings()
        self.client_id = client_id or settings.reddit_client_id
        self.client_secret = client_secret or settings.reddit_client_secret
        self.user_agent = user_agent or settings.reddit_user_agent

    def fetch_posts(
        self,
        symbols: List[str],
        subreddits: List[str] = ["wallstreetbets", "stocks", "investing"],
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch top and new submissions mentioning the given ticker symbols.
        """
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials missing; generating synthetic sentiment text.")
            return self._generate_synthetic_posts(symbols, limit)

        try:
            import praw
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )

            records = []
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.new(limit=limit):
                    text_content = f"{post.title} {post.selftext}"
                    for symbol in symbols:
                        if symbol in text_content or f"${symbol}" in text_content:
                            records.append({
                                "id": post.id,
                                "timestamp": datetime.fromtimestamp(post.created_utc, timezone.utc),
                                "symbol": symbol,
                                "source": f"reddit/r/{sub_name}",
                                "text": post.title,
                                "score": post.score,
                                "num_comments": post.num_comments,
                            })

            df = pd.DataFrame(records)
            if df.empty:
                return pd.DataFrame(columns=["id", "timestamp", "symbol", "source", "text", "score", "num_comments"])
            return df
        except Exception as e:
            logger.error(f"Failed to scrape Reddit: {str(e)}")
            raise DataIngestionError(f"Reddit scraper error: {str(e)}")

    def _generate_synthetic_posts(self, symbols: List[str], count: int) -> pd.DataFrame:
        """Generate realistic synthetic Reddit posts for testing."""
        import random
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
            records.append({
                "id": f"reddit_syn_{i}",
                "timestamp": now - random.random() * random.randint(1, 300) * pd.Timedelta(minutes=1),
                "symbol": symbol,
                "source": "reddit/r/wallstreetbets",
                "text": f"${symbol} - {random.choice(headlines)}",
                "score": random.randint(5, 1200),
                "num_comments": random.randint(2, 450),
            })
        return pd.DataFrame(records)

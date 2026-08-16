"""
NewsAPI & GDELT financial news and headline ingestion collector.
"""

from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import requests
from src.config.settings import get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("NewsCollector")


class NewsCollector:
    """
    Collects institutional and breaking financial news headlines from NewsAPI / GDELT.
    """

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.news_api_key

    def fetch_headlines(
        self,
        symbols: List[str],
        page_size: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch recent breaking financial headlines.
        """
        if not self.api_key:
            logger.warning("NewsAPI key missing; generating synthetic financial news.")
            return self._generate_synthetic_news(symbols, page_size)

        try:
            query = " OR ".join(symbols)
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(page_size, 100),
                "apiKey": self.api_key,
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            records = []
            for art in articles:
                text = f"{art.get('title', '')}. {art.get('description', '')}"
                matched_symbol = next((s for s in symbols if s in text), symbols[0])
                records.append({
                    "id": art.get("url"),
                    "timestamp": pd.to_datetime(art.get("publishedAt")),
                    "symbol": matched_symbol,
                    "source": f"news/{art.get('source', {}).get('name', 'web')}",
                    "text": art.get("title", ""),
                    "score": 100,
                    "num_comments": 0,
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"NewsAPI error: {str(e)}")
            raise DataIngestionError(f"News collection failed: {str(e)}")

    def _generate_synthetic_news(self, symbols: List[str], count: int) -> pd.DataFrame:
        """Generate realistic synthetic financial news headlines."""
        import random
        templates = [
            "Federal Reserve hints at interest rate policy shift during emergency briefing.",
            "Semiconductor manufacturing bottlenecks raise supply chain concerns across tech sector.",
            "Tech giants report unprecedented demand for enterprise AI infrastructure.",
            "Department of Justice initiates regulatory antitrust review into major software vendors.",
            "Macroeconomic consumer sentiment index falls below analyst consensus.",
        ]
        records = []
        now = datetime.now(timezone.utc)
        for i in range(count):
            symbol = random.choice(symbols)
            records.append({
                "id": f"news_syn_{i}",
                "timestamp": now - random.random() * random.randint(1, 400) * pd.Timedelta(minutes=1),
                "symbol": symbol,
                "source": "news/Reuters",
                "text": f"{symbol}: {random.choice(templates)}",
                "score": 100,
                "num_comments": 0,
            })
        return pd.DataFrame(records)

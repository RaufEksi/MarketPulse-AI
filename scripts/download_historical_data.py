"""
Standalone script to download historical intraday bars and sentiment datasets.
Usage: python scripts/download_historical_data.py --symbol SPY --days 30
"""

import argparse
from datetime import datetime, timedelta, timezone
from src.config.settings import get_settings
from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.data_engine.reddit_collector import RedditCollector
from src.data_engine.news_collector import NewsCollector
from src.data_engine.storage_manager import StorageManager
from src.utils.logger import get_logger

logger = get_logger("DownloadScript")


def main():
    parser = argparse.ArgumentParser(description="Download historical market and sentiment data.")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol (e.g. SPY, AAPL)")
    parser.add_argument("--days", type=int, default=14, help="Number of historical days to fetch")
    args = parser.parse_args()

    settings = get_settings()
    storage = StorageManager()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.days)

    logger.info(f"Ingesting 5-min bars for {args.symbol} from {start_time} to {end_time}...")
    alpaca = AlpacaDataCollector()
    bars_df = alpaca.fetch_5min_bars(args.symbol, start_time, end_time)
    storage.save_raw_bars(bars_df, args.symbol)

    logger.info("Ingesting Reddit and News headlines...")
    reddit = RedditCollector()
    reddit_df = reddit.fetch_posts([args.symbol])

    news = NewsCollector()
    news_df = news.fetch_headlines([args.symbol])

    logger.info(f"Downloaded {len(bars_df)} bars, {len(reddit_df)} Reddit posts, and {len(news_df)} news articles.")


if __name__ == "__main__":
    main()

"""
Parquet Data Lake & Local Storage Manager.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("StorageManager")


class StorageManager:
    """
    Manages structured Parquet data lake with partitioning by source, symbol, and date.
    """

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.data.storage_dir)
        self.raw_dir = Path(settings.data.raw_dir)
        self.processed_dir = Path(settings.data.processed_dir)
        self.cache_dir = Path(settings.data.cache_dir)

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for directory in [self.raw_dir, self.processed_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_raw_bars(self, df: pd.DataFrame, symbol: str) -> Path:
        """
        Save raw OHLCV bars partitioned by symbol.
        """
        out_dir = self.raw_dir / "ohlcv" / f"symbol={symbol}"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "bars.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} bars for {symbol} to {file_path}")
        return file_path

    def load_raw_bars(self, symbol: str) -> pd.DataFrame:
        """
        Load raw OHLCV bars for a symbol.
        """
        file_path = self.raw_dir / "ohlcv" / f"symbol={symbol}" / "bars.parquet"
        if not file_path.exists():
            logger.warning(f"No stored bars found at {file_path}")
            return pd.DataFrame()
        return pd.read_parquet(file_path, engine="pyarrow")

    def save_processed_dataset(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save aligned & feature-engineered dataset.
        """
        file_path = self.processed_dir / f"{filename}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        logger.info(f"Saved processed dataset ({len(df)} rows) to {file_path}")
        return file_path

    def load_processed_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load processed dataset from Parquet lake.
        """
        file_path = self.processed_dir / f"{filename}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {file_path}")
        return pd.read_parquet(file_path, engine="pyarrow")

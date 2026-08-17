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

    def save_manifest(self, metadata: dict, manifest_name: Optional[str] = None) -> Path:
        """
        Save dataset ingestion manifest metadata JSON.
        """
        import json
        from datetime import datetime, timezone

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = manifest_name or f"{now_str}_manifest.json"
        manifest_path = self.raw_dir / name
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved dataset manifest to {manifest_path}")
        return manifest_path

    def clean_retention_policy(self, max_age_days: int = 365) -> int:
        """
        Clean up files older than max_age_days rolling window.
        """
        import time

        now_ts = time.time()
        max_age_seconds = max_age_days * 86400
        removed_count = 0

        for path in self.raw_dir.glob("**/*"):
            if path.is_file() and not path.name.endswith("_manifest.json"):
                file_age = now_ts - path.stat().st_mtime
                if file_age > max_age_seconds:
                    path.unlink()
                    removed_count += 1
                    logger.info(f"Retention policy removed expired file: {path}")

        return removed_count

    def save_processed_dataset(
        self, df: pd.DataFrame, filename: str, compression: str = "snappy"
    ) -> Path:
        """
        Save aligned & feature-engineered dataset with PyArrow compression.
        """
        file_path = self.processed_dir / f"{filename}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow", compression=compression)
        logger.info(f"Saved processed dataset ({len(df)} rows, {compression}) to {file_path}")
        return file_path

    def load_processed_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load processed dataset from Parquet lake.
        """
        file_path = self.processed_dir / f"{filename}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {file_path}")
        return pd.read_parquet(file_path, engine="pyarrow")

"""
Pydantic settings and YAML configuration loader for MarketPulse AI.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    name: str = "MarketPulse AI"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    seed: int = 42


class DataSettings(BaseSettings):
    symbols: List[str] = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]
    bar_timeframe: str = "5Min"
    lookback_bars: int = 78
    prediction_horizon_bars: int = 6
    volatility_threshold_pct: float = 0.15
    decay_lambda_per_hour: float = 0.5
    storage_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    reddit_api_base_url: str = "https://arctic-shift.photon-reddit.com"
    reddit_subreddits: List[str] = ["wallstreetbets", "stocks", "investing"]
    reddit_request_timeout_s: float = 15.0
    reddit_use_synthetic: bool = False


class NLPSettings(BaseSettings):
    finbert_model: str = "ProsusAI/finbert"
    max_length: int = 512
    embedding_dim: int = 768
    batch_size: int = 32
    device: str = "cpu"


class TimeSeriesModelSettings(BaseSettings):
    input_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    model_type: str = "bilstm"


class TextNLPModelSettings(BaseSettings):
    input_dim: int = 768
    projected_dim: int = 128
    dropout: float = 0.2


class FusionModelSettings(BaseSettings):
    embed_dim: int = 128
    num_heads: int = 4
    dense_dim: int = 64
    dropout: float = 0.3
    num_classes: int = 2


class ModelSettings(BaseSettings):
    time_series: TimeSeriesModelSettings = TimeSeriesModelSettings()
    text_nlp: TextNLPModelSettings = TextNLPModelSettings()
    fusion: FusionModelSettings = FusionModelSettings()


class TrainingSettings(BaseSettings):
    batch_size: int = 64
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    epochs: int = 50
    early_stopping_patience: int = 7
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    val_split: float = 0.15
    test_split: float = 0.15


class APISettings(BaseSettings):
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000
    reload: bool = False
    workers: int = 2


class DashboardSettings(BaseSettings):
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8501
    theme: str = "dark"
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: float = 120.0
    cache_ttl_seconds: int = 60
    bars_limit: int = 78
    texts_limit: int = 20


class Settings(BaseSettings):
    """
    Unified Application Settings.
    Loads from .env first, then default.yaml, with environment-specific overrides.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Secrets from .env
    alpaca_api_key: Optional[str] = Field(default=None, alias="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )
    news_api_key: Optional[str] = Field(default=None, alias="NEWS_API_KEY")
    # Overrides dashboard.api_base_url (e.g. http://api:8000 inside docker-compose)
    marketpulse_api_url: Optional[str] = Field(default=None, alias="MARKETPULSE_API_URL")

    # Structured YAML Sub-configs
    app: AppSettings = AppSettings()
    data: DataSettings = DataSettings()
    nlp: NLPSettings = NLPSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    api: APISettings = APISettings()
    dashboard: DashboardSettings = DashboardSettings()

    @classmethod
    def load_from_yaml(cls, yaml_path: Path | str = "config/default.yaml") -> "Settings":
        """Load settings from a YAML file with fallback to defaults."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton provider for application settings."""
    return Settings.load_from_yaml()

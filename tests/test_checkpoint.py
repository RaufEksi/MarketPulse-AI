"""
Tests for MarketPulseNet checkpoint serialization and the training script.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data_engine.reddit_collector import RedditCollector
from src.feature_engineering.sentiment_embedder import FinBERTEmbedder
from src.feature_engineering.technical_indicators import MODEL_FEATURE_COLUMNS
from src.models.checkpoint import fit_feature_normalizer, load_checkpoint, save_checkpoint
from src.models.hybrid_network import MarketPulseNet
from src.utils.exceptions import ModelInferenceError

MODEL_CONFIG = {"ts_input_dim": 16, "text_input_dim": 768, "hidden_dim": 32, "num_heads": 4}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _save_small_checkpoint(path: Path) -> MarketPulseNet:
    torch.manual_seed(0)
    model = MarketPulseNet(**MODEL_CONFIG)
    save_checkpoint(
        model,
        path,
        model_config=MODEL_CONFIG,
        feature_columns=MODEL_FEATURE_COLUMNS,
        feature_mean=np.arange(16, dtype=np.float32),
        feature_std=np.full(16, 2.0, dtype=np.float32),
        metadata={"text_embedder": "ProsusAI/finbert"},
    )
    return model


class TestCheckpoint:
    """Tests for save_checkpoint / load_checkpoint."""

    def test_checkpoint_roundtrip_reproduces_outputs(self, tmp_path: Path) -> None:
        """Loaded model gives identical outputs and keeps preprocessing stats."""
        # Arrange
        path = tmp_path / "model.pt"
        original = _save_small_checkpoint(path)
        original.eval()
        ts = torch.randn(2, 78, 16)
        text = torch.randn(2, 768)

        # Act
        bundle = load_checkpoint(path)

        # Assert
        assert not bundle.model.training
        assert bundle.feature_columns == MODEL_FEATURE_COLUMNS
        assert bundle.metadata["text_embedder"] == "ProsusAI/finbert"
        np.testing.assert_allclose(bundle.normalize(np.full((1, 16), 2.0))[0, :2], [1.0, 0.5])
        torch.testing.assert_close(
            bundle.model.predict_probability(ts, text), original.predict_probability(ts, text)
        )

    def test_load_checkpoint_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing checkpoint raises ModelInferenceError."""
        with pytest.raises(ModelInferenceError, match="not found"):
            load_checkpoint(tmp_path / "absent.pt")

    def test_load_checkpoint_rejects_bare_state_dict(self, tmp_path: Path) -> None:
        """A raw state_dict (old trainer output) is rejected with a clear error."""
        path = tmp_path / "raw.pt"
        torch.save(MarketPulseNet(**MODEL_CONFIG).state_dict(), path)
        with pytest.raises(ModelInferenceError, match="format_version"):
            load_checkpoint(path)

    def test_fit_feature_normalizer_floors_constant_std(self) -> None:
        """Constant columns get a non-zero std so normalization never divides by zero."""
        features = np.array([[1.0, 5.0], [3.0, 5.0]])
        mean, std = fit_feature_normalizer(features)
        np.testing.assert_allclose(mean, [2.0, 5.0])
        assert std[0] == pytest.approx(1.0)
        assert std[1] > 0


@pytest.mark.slow
def test_train_script_writes_loadable_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: train_model.py trains on synthetic bars + texts and writes a checkpoint."""
    # Arrange: never download FinBERT in tests
    monkeypatch.setattr(FinBERTEmbedder, "_lazy_load_model", lambda self: False)
    spec = importlib.util.spec_from_file_location(
        "train_model", REPO_ROOT / "scripts" / "train_model.py"
    )
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)
    # Keep the test offline: synthetic Reddit posts instead of Arctic Shift calls
    monkeypatch.setattr(train_model, "RedditCollector", lambda: RedditCollector(use_synthetic=True))
    out = tmp_path / "marketpulse_net.pt"

    # Act
    written = train_model.main(
        [
            "--bars-source=synthetic",
            "--days=3",
            "--epochs=1",
            "--batch-size=64",
            "--allow-fallback-embeddings",
            f"--output={out}",
            f"--checkpoint-dir={tmp_path / 'ckpt'}",
        ]
    )

    # Assert
    bundle = load_checkpoint(written)
    assert written == out
    assert bundle.metadata["text_embedder"] == "hash-fallback"
    assert bundle.metadata["num_texts"] > 0
    assert np.all(bundle.feature_std > 0)


def test_train_script_requires_finbert_by_default(monkeypatch) -> None:
    """Without --allow-fallback-embeddings the script refuses to train on hash embeddings."""
    from src.utils.exceptions import ConfigurationError

    monkeypatch.setattr(FinBERTEmbedder, "_lazy_load_model", lambda self: False)
    spec = importlib.util.spec_from_file_location(
        "train_model", REPO_ROOT / "scripts" / "train_model.py"
    )
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)

    with pytest.raises(ConfigurationError, match="FinBERT"):
        train_model.main(["--bars-source=synthetic", "--days=3"])


def test_load_bars_auto_falls_back_to_yfinance_when_alpaca_fails(monkeypatch) -> None:
    """Auto mode survives Alpaca auth errors (e.g. placeholder keys) by using Yahoo Finance."""
    from src.utils.exceptions import DataIngestionError

    # Arrange
    spec = importlib.util.spec_from_file_location(
        "train_model", REPO_ROOT / "scripts" / "train_model.py"
    )
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)
    settings = train_model.get_settings()
    monkeypatch.setattr(settings, "alpaca_api_key", "placeholder")
    monkeypatch.setattr(settings, "alpaca_secret_key", "placeholder")

    def fail(*args, **kwargs):
        raise DataIngestionError("401 Unauthorized")

    def fake_yfinance(self, symbol, period):
        end = train_model.datetime.now(train_model.timezone.utc)
        start = end - train_model.timedelta(days=1)
        return train_model.AlpacaDataCollector()._generate_synthetic_bars(symbol, start, end)

    monkeypatch.setattr(train_model.AlpacaDataCollector, "fetch_5min_bars", fail)
    monkeypatch.setattr(train_model.YFinanceDataCollector, "fetch_5min_bars", fake_yfinance)

    # Act
    bars = train_model.load_bars("auto", "SPY", days=1)

    # Assert
    assert len(bars) > 0

    # An explicit --bars-source alpaca still surfaces the error
    with pytest.raises(DataIngestionError):
        train_model.load_bars("alpaca", "SPY", days=1)


def test_fetch_reddit_by_day_covers_range_and_skips_failed_days() -> None:
    """Reddit is fetched in daily windows; a failing day is skipped, not fatal."""
    from datetime import datetime, timezone

    import pandas as pd

    from src.utils.exceptions import DataIngestionError

    # Arrange
    spec = importlib.util.spec_from_file_location(
        "train_model", REPO_ROOT / "scripts" / "train_model.py"
    )
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)
    calls = []

    class FakeCollector:
        def fetch_posts(self, symbols, limit, after, before):
            calls.append((after, before))
            if len(calls) == 2:
                raise DataIngestionError("422")
            return pd.DataFrame({"timestamp": [after], "symbol": symbols, "text": ["x"]})

    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)

    # Act
    result = train_model.fetch_reddit_by_day(FakeCollector(), "SPY", start, end, limit=10)

    # Assert
    assert len(calls) == 3
    assert calls[0][0] == start and calls[-1][1] == end
    assert len(result) == 2


def test_load_texts_skips_synthetic_news_without_api_key(monkeypatch) -> None:
    """Without a NewsAPI key, training must not ingest NewsCollector's synthetic headlines."""
    import pandas as pd

    # Arrange
    spec = importlib.util.spec_from_file_location(
        "train_model", REPO_ROOT / "scripts" / "train_model.py"
    )
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)
    monkeypatch.setattr(train_model.get_settings(), "news_api_key", None)
    reddit = pd.DataFrame(
        {"timestamp": ["2026-09-01T14:00:00Z"], "symbol": ["SPY"], "text": ["real post"]}
    )
    monkeypatch.setattr(train_model, "fetch_reddit_by_day", lambda *args, **kwargs: reddit)

    def news_called(*args, **kwargs):
        raise AssertionError("NewsCollector must not be called without an API key")

    monkeypatch.setattr(train_model.NewsCollector, "fetch_headlines", news_called)

    # Act
    texts = train_model.load_texts("SPY", None, use_collectors=True)

    # Assert
    assert texts["text"].tolist() == ["real post"]

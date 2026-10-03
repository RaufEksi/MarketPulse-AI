"""
Tests for the /backtest endpoint: symbol-specific data and model-driven probabilities.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import backtest
from src.config.settings import get_settings
from src.feature_engineering.technical_indicators import MODEL_FEATURE_COLUMNS
from src.models.checkpoint import ModelBundle
from src.utils.exceptions import DataIngestionError, ModelInferenceError

client = TestClient(app)

SEQ_LEN = get_settings().data.lookback_bars
PAYLOAD = {
    "symbol": "SPY",
    "spike_threshold": 0.5,
    "hedge_reduction_factor": 0.2,
    "initial_capital": 100000.0,
}


class ConstantModel:
    """Stub model returning a fixed probability and recording input shapes."""

    def __init__(self, prob: float) -> None:
        self.prob = prob
        self.ts_shapes: List[torch.Size] = []
        self.text_shapes: List[torch.Size] = []

    def predict_probability(self, ts_input: torch.Tensor, text_input: torch.Tensor) -> torch.Tensor:
        self.ts_shapes.append(ts_input.shape)
        self.text_shapes.append(text_input.shape)
        return torch.full((ts_input.shape[0],), self.prob)


def _bundle(model: object, mean: float = 0.0, std: float = 1.0) -> ModelBundle:
    """Wrap a model in a bundle with uniform normalization stats."""
    num_features = len(MODEL_FEATURE_COLUMNS)
    return ModelBundle(
        model=model,  # type: ignore[arg-type]
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        feature_mean=np.full(num_features, mean, dtype=np.float32),
        feature_std=np.full(num_features, std, dtype=np.float32),
    )


def _bars(n: int) -> pd.DataFrame:
    """Build n synthetic 5-min bars."""
    rng = np.random.default_rng(0)
    start = datetime(2026, 9, 1, 13, 30, tzinfo=timezone.utc)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, size=n)))
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": rng.integers(1000, 5000, size=n).astype(float),
        }
    )


class TestBacktestEndpoint:
    """Tests for POST /backtest."""

    def test_backtest_results_vary_by_symbol(self) -> None:
        """Test different symbols are backtested on different price histories."""
        # Act
        spy = client.post("/backtest", json={**PAYLOAD, "symbol": "SPY"}).json()
        aapl = client.post("/backtest", json={**PAYLOAD, "symbol": "aapl"}).json()
        msft = client.post("/backtest", json={**PAYLOAD, "symbol": "MSFT"}).json()

        # Assert
        assert aapl["symbol"] == "AAPL"
        assert spy["benchmark_equity"] != aapl["benchmark_equity"]
        assert aapl["benchmark_equity"] != msft["benchmark_equity"]
        assert spy["is_synthetic"] is True

    def test_backtest_probabilities_come_from_model(self, monkeypatch) -> None:
        """Test every simulated bar is scored by the model with full lookback windows."""
        # Arrange
        model = ConstantModel(prob=0.9)
        monkeypatch.setattr(backtest, "get_backtest_model", lambda: _bundle(model))
        monkeypatch.setattr(backtest, "_load_bars", lambda symbol: (_bars(SEQ_LEN + 50), False))

        # Act
        data = client.post("/backtest", json=PAYLOAD).json()

        # Assert
        assert data["predicted_probabilities"] == pytest.approx([0.9] * 51)
        assert len(data["benchmark_equity"]) == 50
        assert all(shape[1:] == (SEQ_LEN, len(MODEL_FEATURE_COLUMNS)) for shape in model.ts_shapes)
        assert sum(shape[0] for shape in model.ts_shapes) == 51
        assert data["is_synthetic"] is False

    def test_backtest_high_probability_hedges_position(self, monkeypatch) -> None:
        """Test model probabilities above threshold scale exposure to the hedge factor."""
        # Arrange
        monkeypatch.setattr(backtest, "_load_bars", lambda symbol: (_bars(SEQ_LEN + 20), False))
        monkeypatch.setattr(
            backtest, "get_backtest_model", lambda: _bundle(ConstantModel(prob=0.9))
        )

        # Act
        hedged = client.post("/backtest", json=PAYLOAD).json()
        monkeypatch.setattr(
            backtest, "get_backtest_model", lambda: _bundle(ConstantModel(prob=0.1))
        )
        unhedged = client.post("/backtest", json=PAYLOAD).json()

        # Assert
        capital = PAYLOAD["initial_capital"]
        benchmark_step = hedged["benchmark_equity"][0] / capital - 1.0
        hedged_step = hedged["strategy_equity"][0] / capital - 1.0
        assert hedged_step == pytest.approx(benchmark_step * PAYLOAD["hedge_reduction_factor"])
        assert unhedged["strategy_equity"] == pytest.approx(unhedged["benchmark_equity"])

    def test_backtest_connector_failure_returns_502(self, monkeypatch) -> None:
        """Test a data connector error surfaces as a bad gateway response."""

        # Arrange
        def fail(symbol: str):
            raise DataIngestionError("Alpaca API error: timeout")

        monkeypatch.setattr(backtest, "_load_bars", fail)

        # Act
        response = client.post("/backtest", json=PAYLOAD)

        # Assert
        assert response.status_code == 502

    def test_backtest_model_unavailable_returns_503(self, monkeypatch) -> None:
        """Test an unloadable checkpoint surfaces as service unavailable."""

        # Arrange
        def fail():
            raise ModelInferenceError("Could not load checkpoint")

        monkeypatch.setattr(backtest, "_load_bars", lambda symbol: (_bars(SEQ_LEN + 5), False))
        monkeypatch.setattr(backtest, "get_backtest_model", fail)

        # Act
        response = client.post("/backtest", json=PAYLOAD)

        # Assert
        assert response.status_code == 503

    def test_backtest_too_few_bars_returns_422(self, monkeypatch) -> None:
        """Test a history shorter than one window plus one bar is rejected."""
        # Arrange
        monkeypatch.setattr(backtest, "_load_bars", lambda symbol: (_bars(SEQ_LEN), False))

        # Act
        response = client.post("/backtest", json=PAYLOAD)

        # Assert
        assert response.status_code == 422


class TestBuildFeatureWindows:
    """Tests for rolling model input windows."""

    def test_build_feature_windows_no_lookahead(self) -> None:
        """Test each window ends on its own bar and never includes later bars."""
        # Arrange
        df = pd.DataFrame({"close": np.arange(10, dtype=float)})

        # Act
        windows = backtest.build_feature_windows(df, _bundle(None), seq_len=4)

        # Assert
        close_idx = MODEL_FEATURE_COLUMNS.index("close")
        assert windows.shape == (7, 4, len(MODEL_FEATURE_COLUMNS))
        assert windows.dtype == np.float32
        assert windows[0, :, close_idx].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert windows[-1, -1, close_idx] == 9.0
        assert windows[0, 0, MODEL_FEATURE_COLUMNS.index("rsi_14")] == 0.0

    def test_build_feature_windows_applies_normalization(self) -> None:
        """Test windows use the checkpoint's normalization stats."""
        # Arrange
        df = pd.DataFrame({"close": np.arange(10, dtype=float)})

        # Act
        windows = backtest.build_feature_windows(df, _bundle(None, mean=1.0, std=2.0), seq_len=4)

        # Assert
        close_idx = MODEL_FEATURE_COLUMNS.index("close")
        assert windows[0, :, close_idx].tolist() == [-0.5, 0.0, 0.5, 1.0]

    def test_build_feature_windows_short_input_raises(self) -> None:
        """Test fewer rows than one window raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            backtest.build_feature_windows(
                pd.DataFrame({"close": [1.0, 2.0]}), _bundle(None), seq_len=4
            )

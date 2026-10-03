"""
Unit tests for the dashboard REST API client (HTTP calls are mocked).
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.dashboard.components import api_client
from src.utils.exceptions import APIClientError


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Clear cached API responses so each test sees its own mock."""
    for fn in (
        api_client.fetch_health,
        api_client.fetch_bars,
        api_client.fetch_texts,
        api_client.fetch_market_snapshot,
        api_client.run_backtest,
    ):
        fn.clear()
    yield


def _response(status: int, body: object) -> MagicMock:
    resp = MagicMock()
    resp.ok = 200 <= status < 300
    resp.status_code = status
    resp.json.return_value = body
    resp.text = str(body)
    return resp


class TestApiClient:
    """Tests for src.dashboard.components.api_client."""

    def test_fetch_health_calls_health_endpoint(self) -> None:
        """Test fetch_health issues GET /health against the configured base URL."""
        # Arrange
        body = {"status": "healthy"}
        with patch.object(api_client.requests, "request", return_value=_response(200, body)) as req:
            # Act
            result = api_client.fetch_health()

        # Assert
        assert result == body
        method, url = req.call_args.args
        assert method == "GET"
        assert url == f"{api_client.get_api_base_url()}/health"

    def test_request_connection_error_raises_api_client_error(self) -> None:
        """Test an unreachable API becomes APIClientError."""
        with patch.object(
            api_client.requests, "request", side_effect=requests.ConnectionError("refused")
        ):
            with pytest.raises(APIClientError, match="Could not reach"):
                api_client.fetch_health()

    def test_request_http_error_includes_detail(self) -> None:
        """Test a non-2xx response surfaces the FastAPI error detail."""
        resp = _response(400, {"detail": "At least 20 OHLCV bars required"})
        with patch.object(api_client.requests, "request", return_value=resp):
            with pytest.raises(APIClientError, match="20 OHLCV bars"):
                api_client.run_backtest("SPY", 0.65, 0.2, 100000.0)

    def test_fetch_market_snapshot_posts_bars_and_texts_to_predict(self) -> None:
        """Test the snapshot forwards /market data to /predict in PredictRequest shape."""
        # Arrange
        bar = {
            "timestamp": "2026-10-03T14:00:00Z",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100.0,
            "vwap": 1.2,
            "trade_count": 3,
            "atr_14": 0.4,
            "rsi_14": 55.0,
        }
        text = {
            "timestamp": "2026-10-03T13:55:00Z",
            "symbol": "SPY",
            "source": "news/Reuters",
            "text": "Fed holds rates",
            "score": 100.0,
        }
        responses = {
            "/market/bars": {"symbol": "SPY", "is_synthetic": True, "bars": [bar]},
            "/market/texts": {"symbol": "SPY", "texts": [text]},
            "/predict": {"prediction_id": "mp-1"},
        }

        def fake_request(method, url, params=None, json=None, timeout=None):
            path = url.removeprefix(api_client.get_api_base_url())
            return _response(200, responses[path])

        with patch.object(api_client.requests, "request", side_effect=fake_request) as req:
            # Act
            snapshot = api_client.fetch_market_snapshot("SPY")

        # Assert
        assert snapshot["prediction"]["prediction_id"] == "mp-1"
        predict_payload = req.call_args_list[-1].kwargs["json"]
        assert predict_payload["symbol"] == "SPY"
        assert "atr_14" not in predict_payload["ohlcv_bars"][0]
        assert predict_payload["ohlcv_bars"][0]["close"] == 1.5
        assert predict_payload["recent_texts"] == [
            {
                "timestamp": text["timestamp"],
                "headline": "Fed holds rates",
                "source": "news/Reuters",
            }
        ]

    def test_parse_prometheus_text_skips_comments(self) -> None:
        """Test metrics parsing returns one row per sample line."""
        text = "# HELP x y\n# TYPE x counter\nx_total{endpoint='/predict'} 142\n\nlatency 0.018\n"

        rows = api_client.parse_prometheus_text(text)

        assert rows == [
            {"metric": "x_total", "labels": "endpoint='/predict'", "value": 142.0},
            {"metric": "latency", "labels": "", "value": 0.018},
        ]


class TestDashboardLayering:
    """Guards the architecture rule that the dashboard only talks to the API over HTTP."""

    FORBIDDEN = (
        "src.models",
        "src.data_engine",
        "src.feature_engineering",
        "src.data_alignment",
        "src.xai_explainer",
        "src.api",
    )

    def test_dashboard_does_not_import_backend_or_generate_random_data(self) -> None:
        """Test no dashboard file imports backend layers or numpy random generators."""
        from pathlib import Path

        dashboard_dir = Path(api_client.__file__).resolve().parents[1]
        for path in dashboard_dir.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for module in self.FORBIDDEN:
                assert f"from {module}" not in source, f"{path.name} imports {module}"
                assert f"import {module}" not in source, f"{path.name} imports {module}"
            assert "np.random" not in source, f"{path.name} generates random data"
            assert "import random" not in source, f"{path.name} generates random data"

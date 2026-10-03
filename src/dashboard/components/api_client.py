"""
HTTP client for the MarketPulse AI REST API, used by every dashboard page.

The dashboard never imports model, data engine or feature engineering code; all data
comes from the FastAPI service through this module. Responses are cached per session
with Streamlit's data cache so widget interactions do not re-run inference.
"""

from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from src.config.settings import get_settings
from src.utils.exceptions import APIClientError
from src.utils.logger import get_logger

logger = get_logger(__name__)

OHLCV_KEYS = ("timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count")
CACHE_TTL_SECONDS = get_settings().dashboard.cache_ttl_seconds


def get_api_base_url() -> str:
    """
    Resolve the REST API base URL.

    Returns:
        Base URL without a trailing slash; the MARKETPULSE_API_URL env var wins over YAML.
    """
    settings = get_settings()
    return (settings.marketpulse_api_url or settings.dashboard.api_base_url).rstrip("/")


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    """Send one HTTP request to the API and raise APIClientError on any failure."""
    url = f"{get_api_base_url()}{path}"
    timeout = get_settings().dashboard.api_timeout_seconds
    try:
        response = requests.request(method, url, params=params, json=payload, timeout=timeout)
    except requests.RequestException as e:
        logger.error("API unreachable", extra={"url": url, "error": str(e)})
        raise APIClientError(f"Could not reach MarketPulse API at {url}: {e}") from e

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        logger.error("API error", extra={"url": url, "status_code": response.status_code})
        raise APIClientError(
            f"{method} {path} failed with HTTP {response.status_code}: {detail}",
            details={"status_code": response.status_code},
        )
    return response


def _json(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    """Send a request and decode the JSON body."""
    try:
        return _request(method, path, **kwargs).json()
    except ValueError as e:
        raise APIClientError(f"{method} {path} returned invalid JSON") from e


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_health() -> Dict[str, Any]:
    """
    Call GET /health.

    Returns:
        HealthResponse as a dict.

    Raises:
        APIClientError: If the API is unreachable or returns an error.
    """
    return _json("GET", "/health")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_metrics_text() -> str:
    """
    Call GET /metrics.

    Returns:
        Prometheus exposition text.

    Raises:
        APIClientError: If the API is unreachable or returns an error.
    """
    return _request("GET", "/metrics").text


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_bars(symbol: str, limit: int) -> Dict[str, Any]:
    """
    Call GET /market/bars.

    Args:
        symbol: Ticker symbol.
        limit: Number of most recent 5-minute bars.

    Returns:
        MarketBarsResponse as a dict (bars oldest to newest).

    Raises:
        APIClientError: If the API is unreachable or returns an error.
    """
    return _json("GET", "/market/bars", params={"symbol": symbol, "limit": limit})


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_texts(symbol: str, limit: int) -> Dict[str, Any]:
    """
    Call GET /market/texts.

    Args:
        symbol: Ticker symbol.
        limit: Maximum number of texts.

    Returns:
        MarketTextsResponse as a dict (texts newest to oldest).

    Raises:
        APIClientError: If the API is unreachable or returns an error.
    """
    return _json("GET", "/market/texts", params={"symbol": symbol, "limit": limit})


def to_predict_inputs(
    bars: List[Dict[str, Any]],
    texts: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert /market responses into the ohlcv_bars / recent_texts shape /predict expects.

    Args:
        bars: Bars from /market/bars.
        texts: Texts from /market/texts.

    Returns:
        Dict with "ohlcv_bars" and "recent_texts" lists.
    """
    return {
        "ohlcv_bars": [{k: bar.get(k) for k in OHLCV_KEYS} for bar in bars],
        "recent_texts": [
            {"timestamp": t["timestamp"], "headline": t["text"], "source": t["source"]}
            for t in texts
        ],
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_market_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Fetch bars and texts for a symbol and run POST /predict on them.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict with "bars" (MarketBarsResponse), "texts" (MarketTextsResponse) and
        "prediction" (PredictResponse).

    Raises:
        APIClientError: If any call fails.
    """
    settings = get_settings()
    bars = fetch_bars(symbol, settings.dashboard.bars_limit)
    texts = fetch_texts(symbol, settings.dashboard.texts_limit)
    payload = {"symbol": symbol, **to_predict_inputs(bars["bars"], texts["texts"])}
    prediction = _json("POST", "/predict", payload=payload)
    return {"bars": bars, "texts": texts, "prediction": prediction}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_explanation(symbol: str, top_k_features: int) -> Dict[str, Any]:
    """
    Run POST /explain on the same inputs used for the latest /predict call.

    Args:
        symbol: Ticker symbol.
        top_k_features: Number of top features to return.

    Returns:
        Dict with "snapshot" (see fetch_market_snapshot) and "explanation" (ExplainResponse).

    Raises:
        APIClientError: If any call fails.
    """
    snapshot = fetch_market_snapshot(symbol)
    payload = {
        "prediction_id": snapshot["prediction"]["prediction_id"],
        "symbol": symbol,
        "top_k_features": top_k_features,
        **to_predict_inputs(snapshot["bars"]["bars"], snapshot["texts"]["texts"]),
    }
    explanation = _json("POST", "/explain", payload=payload)
    return {"snapshot": snapshot, "explanation": explanation}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def run_backtest(
    symbol: str,
    spike_threshold: float,
    hedge_reduction_factor: float,
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Call POST /backtest.

    Args:
        symbol: Ticker symbol.
        spike_threshold: Probability above which the strategy hedges.
        hedge_reduction_factor: Exposure kept while hedged.
        initial_capital: Starting portfolio value.

    Returns:
        BacktestResponse as a dict.

    Raises:
        APIClientError: If the API is unreachable or returns an error.
    """
    payload = {
        "symbol": symbol,
        "spike_threshold": spike_threshold,
        "hedge_reduction_factor": hedge_reduction_factor,
        "initial_capital": initial_capital,
    }
    return _json("POST", "/backtest", payload=payload)


def parse_prometheus_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse Prometheus exposition text into rows of metric, labels and value.

    Args:
        text: Body of GET /metrics.

    Returns:
        One dict per sample line with keys "metric", "labels" and "value".
    """
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        metric, _, labels = name_part.partition("{")
        try:
            rows.append({"metric": metric, "labels": labels.rstrip("}"), "value": float(value)})
        except ValueError:
            logger.warning("Unparseable metrics line", extra={"line": line})
    return rows


def show_api_error(error: APIClientError) -> None:
    """
    Render an API failure and stop the page.

    Args:
        error: The failure raised by one of the fetch functions.
    """
    st.error(f"MarketPulse API request failed.\n\n{error.message}")
    st.caption(
        f"API base URL: `{get_api_base_url()}`. Start the API with "
        "`python -m uvicorn src.api.main:app` or set MARKETPULSE_API_URL."
    )
    st.stop()

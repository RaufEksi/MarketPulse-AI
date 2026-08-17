"""
Pydantic v2 Request & Response schemas for MarketPulse AI REST API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


class SentimentEvent(BaseModel):
    timestamp: datetime
    headline: str
    source: str = "news"


class PredictRequest(BaseModel):
    symbol: str = Field(..., json_schema_extra={"example": "SPY"})
    ohlcv_bars: List[OHLCVBar]
    recent_texts: List[SentimentEvent] = Field(default_factory=list)


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class PredictResponse(BaseModel):
    prediction_id: str
    symbol: str
    timestamp: datetime
    volatility_spike_probability: float
    risk_level: str  # "LOW_VOLATILITY", "MODERATE_VOLATILITY", "CRITICAL_VOLATILITY"
    confidence_interval: ConfidenceInterval
    inference_latency_ms: float


class ExplainRequest(BaseModel):
    prediction_id: str
    symbol: str = "SPY"
    top_k_features: int = 5
    ohlcv_bars: Optional[List[OHLCVBar]] = None
    recent_texts: Optional[List[SentimentEvent]] = None


class TopFeature(BaseModel):
    feature: str
    shap_value: float


class ExplainResponse(BaseModel):
    prediction_id: str
    symbol: str
    risk_decomposition: Dict[str, Any]
    top_features: List[TopFeature]


class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    spike_threshold: float = 0.65
    hedge_reduction_factor: float = 0.2
    initial_capital: float = 100000.0


class BacktestResponse(BaseModel):
    symbol: str
    strategy_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    strategy_equity: List[float]
    benchmark_equity: List[float]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    active_model: str
    data_pipeline_status: Dict[str, str]

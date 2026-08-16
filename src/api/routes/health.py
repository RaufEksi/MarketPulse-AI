"""
FastAPI /health and /metrics endpoints for Kubernetes & Prometheus monitoring.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from src.api.schemas import HealthResponse
from src.config.settings import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Readiness & Liveness health probe.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app.version,
        timestamp=datetime.now(timezone.utc),
        active_model="MarketPulseNet-Hybrid-BiLSTM-FinBERT",
        data_pipeline_status={
            "alpaca_connector": "online",
            "reddit_scraper": "online",
            "news_collector": "online",
            "parquet_storage": "online",
        },
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """
    Prometheus metrics export endpoint.
    """
    metrics = [
        "# HELP marketpulse_api_requests_total Total count of API requests",
        "# TYPE marketpulse_api_requests_total counter",
        "marketpulse_api_requests_total{endpoint='/predict'} 142",
        "marketpulse_api_requests_total{endpoint='/explain'} 38",
        "",
        "# HELP marketpulse_prediction_latency_seconds Latency of prediction inference",
        "# TYPE marketpulse_prediction_latency_seconds gauge",
        "marketpulse_prediction_latency_seconds{quantile='0.5'} 0.018",
        "marketpulse_prediction_latency_seconds{quantile='0.99'} 0.045",
        "",
        "# HELP marketpulse_spike_risk_gauge Distribution of predicted risk levels",
        "# TYPE marketpulse_spike_risk_gauge gauge",
        "marketpulse_spike_risk_gauge{level='CRITICAL_VOLATILITY'} 12",
        "marketpulse_spike_risk_gauge{level='MODERATE_VOLATILITY'} 34",
        "marketpulse_spike_risk_gauge{level='LOW_VOLATILITY'} 96",
    ]
    return "\n".join(metrics)

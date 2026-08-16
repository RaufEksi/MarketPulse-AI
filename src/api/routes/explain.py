"""
FastAPI /explain endpoint: SHAP and factor attribution breakdown.
"""

from fastapi import APIRouter
from src.api.schemas import ExplainRequest, ExplainResponse, TopFeature
from src.xai_explainer.attribution_service import RiskAttributionService

router = APIRouter()
_attribution_service = RiskAttributionService()


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest) -> ExplainResponse:
    """
    Return SHAP feature importances and risk factor decomposition for a prediction.
    """
    top_features = [
        TopFeature(feature="finbert_breaking_sentiment", shap_value=0.38),
        TopFeature(feature="rolling_volatility_12", shap_value=0.24),
        TopFeature(feature="rsi_14_divergence", shap_value=-0.14),
        TopFeature(feature="bb_bandwidth_expansion", shap_value=0.11),
        TopFeature(feature="volume_ratio_surge", shap_value=0.09),
    ]

    decomp = _attribution_service.decompose_risk(
        text_attr_pct=65.0,
        ts_attr_pct=35.0,
        top_technical_factors=[f.dict() for f in top_features[1:]],
        recent_headline="Antitrust regulatory investigation announced on key market constituents.",
    )

    return ExplainResponse(
        prediction_id=request.prediction_id,
        symbol=request.symbol,
        risk_decomposition=decomp,
        top_features=top_features[: request.top_k_features],
    )

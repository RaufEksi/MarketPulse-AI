"""
Unit tests for Explainable AI (XAI) and Risk Attribution components.
"""

import numpy as np
import torch
import pytest
from src.xai_explainer.attribution_service import RiskAttributionService
from src.xai_explainer.integrated_gradients import IntegratedGradientsExplainer
from src.xai_explainer.shap_explainer import ShapExplainer
from src.models.hybrid_network import MarketPulseNet


def test_risk_attribution_service():
    service = RiskAttributionService()
    decomp = service.decompose_risk(
        text_attr_pct=60.0,
        ts_attr_pct=40.0,
        top_technical_factors=[{"feature": "rsi_14", "shap_value": 0.25}],
        recent_headline="Test breaking news headline",
    )

    assert "news_sentiment_pct" in decomp
    assert "technical_indicators_pct" in decomp
    assert decomp["news_sentiment_pct"] == 60.0
    assert decomp["technical_indicators_pct"] == 40.0
    assert "summary_narrative" in decomp


def test_integrated_gradients():
    model = MarketPulseNet(ts_input_dim=16, text_input_dim=768, hidden_dim=128)
    explainer = IntegratedGradientsExplainer(model)

    ts_sample = torch.randn(1, 78, 16)
    text_sample = torch.randn(1, 768)

    attrs = explainer.attribute(ts_sample, text_sample, steps=5)
    assert "time_series_attribution_pct" in attrs
    assert "sentiment_text_attribution_pct" in attrs
    assert "ts_time_step_importance" in attrs
    assert len(attrs["ts_time_step_importance"]) == 78


def test_shap_explainer():
    class DummyModel:
        def predict_proba(self, x):
            probs = np.zeros((len(x), 2))
            probs[:, 1] = 0.5
            probs[:, 0] = 0.5
            return probs

    model = DummyModel()
    bg_data = np.random.normal(0, 1, size=(20, 4))
    explainer = ShapExplainer(model=model, background_data=bg_data)

    sample = np.random.normal(0, 1, size=4)
    feature_names = ["atr", "rsi", "macd", "vol"]
    results = explainer.explain_instance(sample, feature_names)

    assert len(results) == 4
    assert "feature" in results[0]
    assert "shap_value" in results[0]

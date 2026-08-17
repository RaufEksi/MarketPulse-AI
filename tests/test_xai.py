"""
Unit tests for Explainable AI (XAI) and Risk Attribution components.
"""

import numpy as np
import torch

from src.models.hybrid_network import MarketPulseNet
from src.xai_explainer.attribution_service import RiskAttributionService
from src.xai_explainer.integrated_gradients import IntegratedGradientsExplainer
from src.xai_explainer.shap_explainer import DEFAULT_FEATURE_NAMES, ShapExplainer


def test_risk_attribution_service_news_dominant():
    service = RiskAttributionService()
    decomp = service.decompose_risk(
        text_attr_pct=65.0,
        ts_attr_pct=35.0,
        top_technical_factors=[{"feature": "rsi_14", "shap_value": 0.25}],
        recent_headline="Test breaking news headline",
    )

    assert "news_sentiment_pct" in decomp
    assert "technical_indicators_pct" in decomp
    assert decomp["news_sentiment_pct"] == 65.0
    assert decomp["technical_indicators_pct"] == 35.0
    assert decomp["primary_driver"] == "Breaking News & Social Sentiment (NLP)"
    assert "summary_narrative" in decomp


def test_risk_attribution_service_tech_dominant():
    service = RiskAttributionService()
    decomp = service.decompose_risk(
        text_attr_pct=20.0,
        ts_attr_pct=80.0,
        top_technical_factors=[
            {"feature": "atr_14", "shap_value": 0.35},
            {"feature": "bb_bandwidth", "shap_value": 0.25},
        ],
        recent_headline="",
    )

    assert decomp["technical_indicators_pct"] == 80.0
    assert decomp["primary_driver"] == "Technical Price Action & Order Flow Dynamics"
    assert "atr_14" in decomp["technical_subcomponents"]


def test_integrated_gradients():
    model = MarketPulseNet(ts_input_dim=16, text_input_dim=768, hidden_dim=128)
    explainer = IntegratedGradientsExplainer(model)

    ts_sample = torch.randn(1, 78, 16)
    text_sample = torch.randn(1, 768)

    attrs = explainer.attribute(
        ts_sample, text_sample, steps=5, feature_names=DEFAULT_FEATURE_NAMES
    )
    assert "time_series_attribution_pct" in attrs
    assert "sentiment_text_attribution_pct" in attrs
    assert "ts_time_step_importance" in attrs
    assert len(attrs["ts_time_step_importance"]) == 78
    assert "attention_weights" in attrs
    assert "feature_attributions" in attrs
    assert "atr_14" in attrs["feature_attributions"]


def test_shap_explainer_with_custom_features():
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


def test_shap_explainer_default_features():
    def mock_predict(x):
        return np.mean(x, axis=1)

    explainer = ShapExplainer(model=mock_predict)
    sample = np.random.normal(0, 1, size=16)
    results = explainer.explain_instance(sample)

    assert len(results) == 16
    assert any(r["feature"] == "atr_14" for r in results)

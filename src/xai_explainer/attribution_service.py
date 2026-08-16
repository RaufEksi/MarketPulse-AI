"""
Volatility Risk Factor Decomposition Service.
Produces intuitive percentage breakdown: e.g. 65% Breaking News, 20% RSI Divergence, 15% Volume Surge.
"""

from typing import Dict, Any, List
import numpy as np


class RiskAttributionService:
    """
    Translates mathematical model attributions (SHAP & Integrated Gradients)
    into institutional human-readable factor decomposition cards.
    """

    def decompose_risk(
        self,
        text_attr_pct: float,
        ts_attr_pct: float,
        top_technical_factors: List[Dict[str, Any]],
        recent_headline: str = "",
    ) -> Dict[str, Any]:
        """
        Produce normalized attribution breakdown.
        """
        # Ensure percentages sum to 100%
        total = text_attr_pct + ts_attr_pct + 1e-9
        news_share = (text_attr_pct / total) * 100.0
        tech_share = (ts_attr_pct / total) * 100.0

        # Sub-divide technical share among top technical indicators
        tech_breakdown = {}
        if top_technical_factors:
            tech_total_weight = sum(abs(f.get("shap_value", 1.0)) for f in top_technical_factors) + 1e-9
            for factor in top_technical_factors[:3]:
                fname = factor.get("feature", "technical_indicator")
                weight = abs(factor.get("shap_value", 1.0))
                factor_share = (weight / tech_total_weight) * tech_share
                tech_breakdown[fname] = round(factor_share, 1)

        summary_narrative = (
            f"Model flagged elevated risk primarily driven by {round(news_share)}% Sentiment/News shock "
            f"('{recent_headline[:60]}...') and {round(tech_share)}% Technical market momentum indicators."
        )

        return {
            "news_sentiment_pct": round(news_share, 1),
            "technical_indicators_pct": round(tech_share, 1),
            "technical_subcomponents": tech_breakdown,
            "headline_context": recent_headline,
            "summary_narrative": summary_narrative,
        }

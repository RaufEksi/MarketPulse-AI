"""
Unit tests for configuration loading and validation.
"""

from src.config.settings import Settings, get_settings


def test_settings_default_values():
    settings = get_settings()
    assert settings.app.name == "MarketPulse AI"
    assert "SPY" in settings.data.symbols
    assert settings.data.prediction_horizon_bars == 6
    assert settings.data.volatility_threshold_pct == 0.15
    assert settings.model.time_series.hidden_dim == 128
    assert settings.training.loss_type == "focal"


def test_yaml_config_loading():
    settings = Settings.load_from_yaml("config/default.yaml")
    assert settings.app.seed == 42
    assert settings.nlp.embedding_dim == 768

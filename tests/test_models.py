"""
Unit tests for Machine Learning and Deep Learning architectures.
"""

import numpy as np
import torch
from src.models.baseline_models import BaselineModelTrainer
from src.models.hybrid_network import MarketPulseNet
from src.models.loss_functions import BinaryFocalLoss
from src.models.backtester import VolatilityBacktester


def test_baseline_model_trainer():
    X = np.random.normal(0, 1, size=(100, 16))
    y = np.random.choice([0, 1], size=100)

    trainer = BaselineModelTrainer("hist_gb")
    trainer.fit(X, y)
    probs = trainer.predict_proba(X)

    assert len(probs) == 100
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_marketpulse_net_forward_pass():
    batch_size = 4
    seq_len = 78
    ts_dim = 16
    text_dim = 768

    model = MarketPulseNet(ts_input_dim=ts_dim, text_input_dim=text_dim, hidden_dim=128)
    ts_input = torch.randn(batch_size, seq_len, ts_dim)
    text_input = torch.randn(batch_size, text_dim)

    logits, attn_weights = model(ts_input, text_input)
    assert logits.shape == (batch_size,)
    assert attn_weights.shape == (batch_size, 1, seq_len)

    probs = model.predict_probability(ts_input, text_input)
    assert probs.shape == (batch_size,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_binary_focal_loss():
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.tensor([2.0, -1.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss = criterion(logits, targets)
    assert loss.item() > 0.0


def test_volatility_backtester():
    prices = np.array([100.0, 101.0, 102.0, 98.0, 99.0, 105.0])
    probs = np.array([0.1, 0.2, 0.8, 0.85, 0.2, 0.1])

    backtester = VolatilityBacktester(spike_threshold=0.65)
    results = backtester.run(prices, probs, initial_capital=10000.0)

    assert "strategy_metrics" in results
    assert "benchmark_metrics" in results
    assert len(results["strategy_equity"]) == len(prices) - 1

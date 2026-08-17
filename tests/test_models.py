"""
Unit tests for Machine Learning and Deep Learning architectures.
"""

import numpy as np
import torch

from src.models.backtester import VolatilityBacktester
from src.models.baseline_models import BaselineModelTrainer
from src.models.hybrid_network import MarketPulseNet
from src.models.loss_functions import BinaryFocalLoss


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


def test_time_series_encoder_tcn():
    from src.models.time_series_branch import TimeSeriesEncoder

    tcn_encoder = TimeSeriesEncoder(input_dim=16, hidden_dim=128, model_type="tcn")
    x = torch.randn(4, 78, 16)
    out = tcn_encoder(x)
    assert out.shape == (4, 78, 128)


def test_random_forest_baseline():
    X = np.random.normal(0, 1, size=(50, 16))
    y = np.random.choice([0, 1], size=50)

    trainer = BaselineModelTrainer("random_forest")
    trainer.fit(X, y)
    metrics = trainer.evaluate(X, y)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics


def test_model_trainer_fit(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.trainer import ModelTrainer

    model = MarketPulseNet(ts_input_dim=16, text_input_dim=768, hidden_dim=64)
    trainer = ModelTrainer(
        model=model, learning_rate=0.001, early_stopping_patience=2, device="cpu"
    )

    ts_data = torch.randn(32, 78, 16)
    text_data = torch.randn(32, 768)
    y_data = torch.randint(0, 2, (32,)).float()

    dataset = TensorDataset(ts_data, text_data, y_data)
    loader = DataLoader(dataset, batch_size=16)

    results = trainer.fit(
        train_loader=loader, val_loader=loader, epochs=3, checkpoint_dir=str(tmp_path)
    )
    assert "best_pr_auc" in results
    assert "best_epoch" in results

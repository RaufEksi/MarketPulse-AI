"""
Standalone training script for MarketPulse AI multi-modal hybrid model.
Usage: python scripts/train_model.py --epochs 20 --batch-size 32
"""

import argparse
import numpy as np
import torch
from src.config.settings import get_settings
from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine
from src.feature_engineering.labeler import VolatilityLabeler
from src.data_alignment.dataset_builder import build_sliding_windows, create_walk_forward_dataloaders
from src.models.hybrid_network import MarketPulseNet
from src.models.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("TrainScript")


def main():
    parser = argparse.ArgumentParser(description="Train MarketPulseNet multi-modal model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--symbol", type=str, default="SPY", help="Asset ticker")
    args = parser.parse_args()

    settings = get_settings()

    # 1. Fetch / Generate data
    from datetime import datetime, timedelta, timezone
    collector = AlpacaDataCollector()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=10)
    bars_df = collector.fetch_5min_bars(args.symbol, start_time, end_time)

    # 2. Feature Engineering & Labels
    feat_engine = TechnicalFeatureEngine()
    features_df = feat_engine.transform(bars_df)

    labeler = VolatilityLabeler()
    labeled_df = labeler.create_labels(features_df)

    # 3. Build features & dummy text embeddings for training demo
    feature_cols = [
        "open", "high", "low", "close", "volume_ratio", "log_return",
        "atr_14", "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "rolling_vol_12", "rolling_vol_36", "rolling_vol_78"
    ]
    feat_mat = labeled_df[feature_cols].values
    text_mat = np.random.normal(0, 0.1, size=(len(labeled_df), 768)).astype(np.float32)
    targets = labeled_df["atr_spike_target"].values

    ts_win, text_win, y_win = build_sliding_windows(feat_mat, text_mat, targets, sequence_length=78)

    train_loader, val_loader, test_loader = create_walk_forward_dataloaders(
        ts_win, text_win, y_win, batch_size=args.batch_size
    )

    # 4. Model & Training
    model = MarketPulseNet(ts_input_dim=16, text_input_dim=768, hidden_dim=128)
    trainer = ModelTrainer(model=model, learning_rate=0.0003, device="cpu")

    results = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    logger.info(f"Training completed successfully. Best Validation PR-AUC: {results['best_pr_auc']:.4f}")


if __name__ == "__main__":
    main()

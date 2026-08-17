"""
PyTorch Training Loop with Focal Loss, LR scheduling, Early Stopping & Checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config.settings import get_settings
from src.models.hybrid_network import MarketPulseNet
from src.models.loss_functions import BinaryFocalLoss
from src.utils.logger import get_logger

logger = get_logger("ModelTrainer")


class ModelTrainer:
    """
    Manages end-to-end model training, validation, metric evaluation,
    early stopping, and model serialization.
    """

    def __init__(
        self,
        model: MarketPulseNet,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        device: str = "cpu",
        early_stopping_patience: Optional[int] = None,
    ):
        settings = get_settings()
        self.model = model.to(device)
        self.device = device
        self.early_stopping_patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else settings.training.early_stopping_patience
        )
        self.criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=settings.training.epochs, eta_min=1e-6
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0

        for ts_batch, text_batch, y_batch in train_loader:
            ts_batch = ts_batch.to(self.device)
            text_batch = text_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(ts_batch, text_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)

        self.scheduler.step()
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate validation or test loader."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for ts_batch, text_batch, y_batch in val_loader:
                ts_batch = ts_batch.to(self.device)
                text_batch = text_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits, _ = self.model(ts_batch, text_batch)
                loss = self.criterion(logits, y_batch)
                probs = torch.sigmoid(logits)

                total_loss += loss.item() * len(y_batch)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        probs_np = np.array(all_probs)
        targets_np = np.array(all_targets)

        try:
            pr_auc = float(average_precision_score(targets_np, probs_np))
        except Exception:
            pr_auc = 0.0

        try:
            roc_auc = float(roc_auc_score(targets_np, probs_np))
        except Exception:
            roc_auc = 0.5

        return {
            "loss": avg_loss,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_dir: str = "models/checkpoints",
    ) -> Dict[str, Any]:
        """
        Execute full training loop with checkpointing and early stopping.
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_pr_auc = -1.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f}"
            )

            if val_metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = val_metrics["pr_auc"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    self.model.state_dict(), checkpoint_path / "best_marketpulse_net.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement for {self.early_stopping_patience} consecutive epochs."
                    )
                    break

        logger.info(f"Training completed. Best PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}")
        return {"best_pr_auc": best_pr_auc, "best_epoch": best_epoch}

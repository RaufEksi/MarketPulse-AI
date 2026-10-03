"""
Self-describing MarketPulseNet checkpoint serialization.

A checkpoint bundles everything inference needs to reproduce the training-time
input pipeline: the model weights, the constructor arguments, the ordered
feature columns, the per-feature normalization statistics fitted on the training
split, and which text embedder produced the training sentiment vectors.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.models.hybrid_network import MarketPulseNet
from src.utils.exceptions import ModelInferenceError
from src.utils.logger import get_logger

logger = get_logger("ModelCheckpoint")

CHECKPOINT_FORMAT_VERSION = 1
MIN_FEATURE_STD = 1e-8  # Guards against division by zero for constant features


@dataclass
class ModelBundle:
    """A loaded MarketPulseNet together with its input preprocessing contract."""

    model: MarketPulseNet
    feature_columns: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply the training-time z-score normalization to raw feature rows.

        Args:
            features: Raw feature array [..., num_features].

        Returns:
            Normalized float32 array with the same shape.
        """
        normalized: np.ndarray = (features - self.feature_mean) / self.feature_std
        return normalized.astype(np.float32)


def fit_feature_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and standard deviation for z-score normalization.

    Args:
        features: Training feature rows [N, num_features].

    Returns:
        Tuple of (mean, std) arrays, each [num_features]; std is floored at MIN_FEATURE_STD.

    Raises:
        ValueError: If ``features`` is empty or not two-dimensional.
    """
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Normalizer needs a non-empty [N, num_features] array.")
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), MIN_FEATURE_STD).astype(np.float32)
    return mean, std


def save_checkpoint(
    model: MarketPulseNet,
    path: Path | str,
    model_config: Dict[str, Any],
    feature_columns: List[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Serialize a trained model and its preprocessing contract to disk.

    Args:
        model: Trained MarketPulseNet.
        path: Destination file path; parent directories are created.
        model_config: Keyword arguments used to construct ``model``.
        feature_columns: Ordered names of the time-series input features.
        feature_mean: Per-feature normalization mean [num_features].
        feature_std: Per-feature normalization std [num_features].
        metadata: Extra JSON-friendly info (embedder, metrics, data span).

    Returns:
        The path the checkpoint was written to.

    Raises:
        ValueError: If the feature statistics do not match ``feature_columns``.
    """
    if not (len(feature_columns) == len(feature_mean) == len(feature_std)):
        raise ValueError("feature_columns, feature_mean and feature_std must have equal length.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": dict(model_config),
        "state_dict": model.state_dict(),
        "feature_columns": list(feature_columns),
        "feature_mean": torch.tensor(np.asarray(feature_mean), dtype=torch.float32),
        "feature_std": torch.tensor(np.asarray(feature_std), dtype=torch.float32),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, out_path)
    logger.info(f"Saved MarketPulseNet checkpoint to {out_path}")
    return out_path


def load_checkpoint(path: Path | str, device: str = "cpu") -> ModelBundle:
    """
    Load a checkpoint written by :func:`save_checkpoint` into an eval-mode model.

    Args:
        path: Checkpoint file path.
        device: Torch device to map weights onto.

    Returns:
        ModelBundle with the restored model and preprocessing statistics.

    Raises:
        ModelInferenceError: If the file is missing, unreadable or incompatible.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise ModelInferenceError(f"Checkpoint not found: {ckpt_path}")

    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
        version = payload.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format_version {version!r}")
        model = MarketPulseNet(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
    except ModelInferenceError:
        raise
    except Exception as e:
        raise ModelInferenceError(f"Could not load checkpoint {ckpt_path}: {e}") from e

    model.to(device)
    model.eval()
    feature_columns = list(payload["feature_columns"])
    feature_mean = payload["feature_mean"].cpu().numpy()
    feature_std = payload["feature_std"].cpu().numpy()
    if not (len(feature_columns) == len(feature_mean) == len(feature_std)):
        raise ModelInferenceError(f"Checkpoint {ckpt_path} has inconsistent feature statistics.")

    logger.info(f"Loaded MarketPulseNet checkpoint from {ckpt_path}")
    return ModelBundle(
        model=model,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata=dict(payload.get("metadata", {})),
    )

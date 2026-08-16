"""
Modeling & Deep Learning Architecture package for MarketPulse AI.
"""

from src.models.baseline_models import BaselineModelTrainer
from src.models.time_series_branch import TimeSeriesEncoder
from src.models.text_branch import TextProjectionEncoder
from src.models.cross_attention import CrossAttentionFusion
from src.models.hybrid_network import MarketPulseNet
from src.models.loss_functions import BinaryFocalLoss
from src.models.trainer import ModelTrainer
from src.models.backtester import VolatilityBacktester

__all__ = [
    "BaselineModelTrainer",
    "TimeSeriesEncoder",
    "TextProjectionEncoder",
    "CrossAttentionFusion",
    "MarketPulseNet",
    "BinaryFocalLoss",
    "ModelTrainer",
    "VolatilityBacktester",
]

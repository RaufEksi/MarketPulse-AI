"""
ATR Volatility Spike Ground-Truth Labeling Generator.
Defines binary target: spike occurs if max(ATR[t+1:t+horizon]) >= ATR[t] * threshold_mult.
"""

import numpy as np
import pandas as pd
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("VolatilityLabeler")


class VolatilityLabeler:
    """
    Computes ground truth volatility spike targets without lookahead leak.
    """

    def __init__(
        self,
        horizon_bars: int = 6,  # 30 minutes for 5-min bars
        threshold_pct: float = 0.15,  # 15% ATR increase
    ):
        settings = get_settings()
        self.horizon_bars = horizon_bars or settings.data.prediction_horizon_bars
        self.threshold_pct = threshold_pct or settings.data.volatility_threshold_pct
        self.multiplier = 1.0 + self.threshold_pct

    def create_labels(self, df: pd.DataFrame, atr_col: str = "atr_14") -> pd.DataFrame:
        """
        Add 'future_max_atr', 'atr_spike_target' (0 or 1), and 'volatility_expansion_ratio' columns.
        """
        df = df.copy()
        if atr_col not in df.columns:
            raise ValueError(f"ATR column '{atr_col}' not found in dataframe.")

        # Compute forward max ATR over horizon
        # Note: rolling window on reversed series or using indexing
        atr_series = df[atr_col].values
        n = len(atr_series)
        future_max = np.full(n, np.nan)

        for t in range(n - self.horizon_bars):
            future_max[t] = np.max(atr_series[t + 1 : t + 1 + self.horizon_bars])

        df["future_max_atr"] = future_max
        df["volatility_expansion_ratio"] = df["future_max_atr"] / (df[atr_col] + 1e-9)

        # Binary label
        df["atr_spike_target"] = (
            df["future_max_atr"] >= (df[atr_col] * self.multiplier)
        ).astype(float)

        # Mark trailing un-evaluable bars as NaN
        df.loc[df["future_max_atr"].isna(), "atr_spike_target"] = np.nan

        spike_pct = (df["atr_spike_target"] == 1.0).mean() * 100.0
        logger.info(f"Generated volatility spike labels. Spike prevalence: {spike_pct:.2f}%")

        return df

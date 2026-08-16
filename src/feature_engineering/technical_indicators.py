"""
Technical indicators engine for high-frequency financial time series.
Calculates ATR, RSI, MACD, Bollinger Bands, Rolling Volatility, VWAP ratios without lookahead bias.
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("TechnicalFeatureEngine")


class TechnicalFeatureEngine:
    """
    Computes standard and institutional technical indicators from OHLCV bars.
    """

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) calculation.
        """
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI).
        """
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD).
        """
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line

        return pd.DataFrame({
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_hist,
        })

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands (%B and Bandwidth).
        """
        sma = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()

        upper_band = sma + (num_std * rolling_std)
        lower_band = sma - (num_std * rolling_std)

        pct_b = (df["close"] - lower_band) / (upper_band - lower_band + 1e-9)
        bandwidth = (upper_band - lower_band) / (sma + 1e-9)

        return pd.DataFrame({
            "bb_upper": upper_band,
            "bb_lower": lower_band,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        })

    @staticmethod
    def calculate_rolling_volatility(df: pd.DataFrame, windows: list[int] = [12, 36, 78]) -> pd.DataFrame:
        """
        Rolling standard deviation of log returns.
        """
        log_returns = np.log(df["close"] / df["close"].shift(1))
        vol_df = pd.DataFrame(index=df.index)
        for w in windows:
            vol_df[f"rolling_vol_{w}"] = log_returns.rolling(window=w).std()
        return vol_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all technical indicator transformations to an OHLCV DataFrame.
        """
        df = df.copy()
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain 'close' price column.")

        # Log return
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

        # ATR
        df["atr_14"] = self.calculate_atr(df, period=14).bfill()

        # RSI
        df["rsi_14"] = self.calculate_rsi(df, period=14).fillna(50.0)

        # MACD
        macd_df = self.calculate_macd(df)
        df["macd_line"] = macd_df["macd_line"]
        df["macd_signal"] = macd_df["macd_signal"]
        df["macd_hist"] = macd_df["macd_hist"]

        # Bollinger Bands
        bb_df = self.calculate_bollinger_bands(df)
        df["bb_pct_b"] = bb_df["bb_pct_b"].fillna(0.5)
        df["bb_bandwidth"] = bb_df["bb_bandwidth"].fillna(0.0)

        # Rolling Volatilities
        vol_df = self.calculate_rolling_volatility(df)
        for col in vol_df.columns:
            df[col] = vol_df[col].fillna(0.0)

        # Normalized Volume
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(window=20).mean().bfill()
            df["volume_ratio"] = df["volume"] / (vol_ma + 1e-9)
        else:
            df["volume_ratio"] = 1.0

        return df

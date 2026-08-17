# Kodlama Standartları — MarketPulse AI

> Bu kurallar, AI IDE'nin ürettiği her kod parçasında tutarlılık sağlamak için uygulanır.

---

## 1. Fonksiyon İmza Standardı

Tüm public fonksiyonlarda type hint ve Google-style docstring **zorunludur**.

```python
def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) for OHLCV data.

    Args:
        highs: Array of high prices [N].
        lows: Array of low prices [N].
        closes: Array of close prices [N].
        period: EMA smoothing period (default: 14).

    Returns:
        Array of ATR values [N].

    Raises:
        ValueError: If arrays have mismatched lengths or period > len(data).
    """
```

**Kurallar**:
- Parametreler her satırda bir tane (trailing comma ile)
- Return type her zaman belirtilmeli (`-> None` dahil)
- Docstring'de Args, Returns, Raises bölümleri olmalı
- Private fonksiyonlarda (`_` ile başlayan) kısa docstring yeterli

---

## 2. Exception Handling Standardı

Projede tanımlı exception hiyerarşisini kullan. Ham `Exception` raise etme.

```python
# ✅ DOĞRU
from src.utils.exceptions import DataIngestionError

def fetch_bars(symbol: str) -> pd.DataFrame:
    try:
        response = client.get_bars(symbol)
    except ConnectionError as e:
        raise DataIngestionError(
            f"Alpaca API bağlantı hatası: {symbol}"
        ) from e

# ❌ YANLIŞ
raise Exception("bir şeyler ters gitti")
```

**Exception hiyerarşisi** (`src/utils/exceptions.py`):
- `MarketPulseException` — base exception
  - `DataIngestionError` — veri çekme hataları
  - `FeatureEngineeringError` — feature hesaplama hataları
  - `ModelInferenceError` — model tahmin hataları
  - `ConfigurationError` — konfigürasyon hataları

---

## 3. Logging Standardı

`print()` kullanma. Structured JSON logger kullan.

```python
# ✅ DOĞRU
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Bars fetched", extra={"symbol": symbol, "count": len(df)})
logger.error("API error", extra={"symbol": symbol, "status_code": 429})

# ❌ YANLIŞ
print(f"Fetched {len(df)} bars for {symbol}")
```

**Kurallar**:
- Her modülde `logger = get_logger(__name__)` ile logger oluştur
- Hassas bilgileri (API key, secret) ASLA loglamaz
- Log seviyeleri: `DEBUG` (geliştirme detayları), `INFO` (normal akış), `WARNING` (potansiyel sorun), `ERROR` (hata), `CRITICAL` (sistem çökmesi)

---

## 4. Config Erişim Standardı

Konfigürasyona doğrudan `.env` okuma veya hardcode ile değil, `get_settings()` ile eriş.

```python
# ✅ DOĞRU
from src.config.settings import get_settings

settings = get_settings()
api_key = settings.alpaca_api_key
symbols = settings.data.symbols
batch_size = settings.training.batch_size

# ❌ YANLIŞ
import os
api_key = os.getenv("ALPACA_API_KEY")

# ❌ YANLIŞ
api_key = "pk_live_abc123"  # hardcode
```

---

## 5. DataFrame Validation Standardı

Veri pipeline'ında DataFrame döndüren fonksiyonlar, çıktıyı validate etmeli.

```python
def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca."""
    df = _raw_fetch(symbol, start, end)

    # Column validation
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise DataIngestionError(f"Missing columns: {missing}")

    # Dtype validation
    assert df["close"].dtype in [np.float32, np.float64], "close must be float"

    # NaN check
    nan_pct = df[required_cols].isna().mean()
    if (nan_pct > 0.1).any():
        logger.warning("High NaN ratio detected", extra={"nan_pct": nan_pct.to_dict()})

    return df
```

---

## 6. PyTorch Model Standardı

```python
import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    """Bidirectional LSTM encoder for price sequences."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // (2 if bidirectional else 1),
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim].

        Returns:
            Encoded representation [batch, seq_len, hidden_dim].
        """
        output, _ = self.lstm(x)
        return self.layer_norm(output)
```

**Kurallar**:
- `__init__`'de tüm layer'lar tanımlanır, `forward`'da iş mantığı
- Device handling: Tensor'ları model'in device'ına taşı (`x.to(self.device)`)
- `batch_first=True` standardı
- Dropout ve LayerNorm her encoder'da olmalı
- Modeller `config/default.yaml`'daki hiperparametreleri `get_settings()` ile almalı

---

## 7. Test Yazma Standardı

Arrange-Act-Assert (AAA) pattern:

```python
import pytest
import numpy as np
from src.feature_engineering.technical_indicators import TechnicalFeatureEngine

class TestATRCalculation:
    """Tests for ATR (Average True Range) computation."""

    def test_atr_basic_computation(self) -> None:
        """Test ATR produces correct shape and non-negative values."""
        # Arrange
        engine = TechnicalFeatureEngine()
        highs = np.array([100, 101, 102, 101, 100], dtype=np.float64)
        lows = np.array([99, 100, 101, 100, 99], dtype=np.float64)
        closes = np.array([99.5, 100.5, 101.5, 100.5, 99.5], dtype=np.float64)

        # Act
        atr = engine.compute_atr(highs, lows, closes, period=2)

        # Assert
        assert atr.shape == highs.shape
        assert (atr[2:] >= 0).all()  # First values may be NaN due to lookback

    def test_atr_insufficient_data_raises(self) -> None:
        """Test ATR raises ValueError when period > data length."""
        engine = TechnicalFeatureEngine()
        with pytest.raises(ValueError, match="period"):
            engine.compute_atr(
                np.array([100.0]), np.array([99.0]), np.array([99.5]),
                period=14
            )

    @pytest.mark.slow
    def test_atr_large_dataset_performance(self) -> None:
        """Test ATR completes within 100ms for 100k rows."""
        # Performance benchmark test
        ...
```

**Kurallar**:
- Sınıf adı `Test<Modül>` formatında
- Metod adı `test_<fonksiyon>_<senaryo>` formatında
- Her testte docstring ile ne test edildiği belirtilmeli
- External API'ler `pytest.fixture` ve `unittest.mock` ile mock'lanmalı
- Fixture'lar `tests/conftest.py`'de tanımlanmalı

---

## 8. Import Sıralama Standardı

isort profil "black" kullanılır:

```python
# 1. Standart kütüphane
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 2. Third-party kütüphaneler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

# 3. Proje-içi modüller
from src.config.settings import get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger
```

---

## 9. Dosya Başlık Standardı

Her Python dosyası modül docstring ile başlamalı:

```python
"""
Alpaca Markets API connector for 5-minute OHLCV bar ingestion.

This module provides the AlpacaDataCollector class that fetches
historical and real-time intraday price data with automatic
rate limiting and retry logic.
"""
```

---

## 10. Constant & Magic Number Standardı

Magic number kullanma. Config veya module-level constant kullan.

```python
# ✅ DOĞRU
ATR_PERIOD = 14
SPIKE_THRESHOLD = 0.15  # 15% ATR increase
LOOKBACK_BARS = 78  # One full trading session (6.5 hours × 12 bars/hour)

def is_spike(current_atr: float, future_atrs: np.ndarray) -> bool:
    return future_atrs.max() >= current_atr * (1 + SPIKE_THRESHOLD)

# ❌ YANLIŞ
def is_spike(current_atr: float, future_atrs: np.ndarray) -> bool:
    return future_atrs.max() >= current_atr * 1.15  # magic number
```

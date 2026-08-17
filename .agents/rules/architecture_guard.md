# Mimari Koruma Kuralları — MarketPulse AI

> Bu kurallar, projenin katmanlı mimarisini korumak, modüller arası yasak bağımlılıkları engellemek ve yeni kod eklerken doğru konumlandırmayı sağlamak için uygulanır.

---

## 1. Katman Bağımlılık Grafiği

Bağımlılık yönü yukarıdan aşağıya doğru. **Tersi yasak.**

```
┌─────────────────────────────────────┐
│          src/dashboard/             │  ← Sadece API'ye HTTP isteği yapar
│        (Streamlit Frontend)         │
├─────────────────────────────────────┤
│            src/api/                 │  ← models, xai, config import edebilir
│       (FastAPI Backend)             │
├─────────────────────────────────────┤
│        src/xai_explainer/           │  ← models import edebilir
│     (SHAP, Integrated Gradients)    │
├─────────────────────────────────────┤
│           src/models/               │  ← data_alignment, feature_eng, config import edebilir
│   (Bi-LSTM, TCN, MarketPulseNet)    │
├─────────────────────────────────────┤
│       src/data_alignment/           │  ← feature_engineering, config import edebilir
│    (Temporal Alignment, Dataset)    │
├─────────────────────────────────────┤
│     src/feature_engineering/        │  ← data_engine, config import edebilir
│  (Teknik göstergeler, FinBERT)      │
├─────────────────────────────────────┤
│         src/data_engine/            │  ← Sadece config ve utils import edebilir
│   (Alpaca, Reddit, NewsAPI, Storage)│
├─────────────────────────────────────┤
│     src/config/ & src/utils/        │  ← Hiçbir üst katmanı import EDEMEZ
│  (Settings, Logger, Exceptions)     │
└─────────────────────────────────────┘
```

---

## 2. Yasak Bağımlılıklar (MUTLAK KURAL)

Aşağıdaki import'lar **hiçbir koşulda** yapılmamalıdır:

| Kaynak Modül | Yasak Import | Neden |
|---|---|---|
| `src/dashboard/` | `from src.models import ...` | Dashboard, modele doğrudan erişmez; API üzerinden gider |
| `src/dashboard/` | `from src.data_engine import ...` | Dashboard, ham veri çekmez; API üzerinden gider |
| `src/dashboard/` | `from src.xai_explainer import ...` | Dashboard, XAI'ye doğrudan erişmez; `/explain` endpoint'i kullanır |
| `src/data_engine/` | `from src.models import ...` | Data engine, model bilmez |
| `src/data_engine/` | `from src.feature_engineering import ...` | Data engine, feature hesaplamaz |
| `src/config/` | `from src.* import ...` (config/utils hariç) | Config katmanı tamamen bağımsız |
| `src/utils/` | `from src.* import ...` (utils hariç) | Utils katmanı tamamen bağımsız |

---

## 3. İzin Verilen Bağımlılıklar (Referans Matris)

| Modül | İzin Verilen Import'lar |
|---|---|
| `src/config/` | stdlib, pydantic, pyyaml |
| `src/utils/` | stdlib, logging, prometheus_client |
| `src/data_engine/` | `src.config`, `src.utils`, pandas, alpaca, praw, requests, pyarrow |
| `src/feature_engineering/` | `src.config`, `src.utils`, `src.data_engine` (storage_manager), pandas, numpy, ta, transformers |
| `src/data_alignment/` | `src.config`, `src.utils`, `src.feature_engineering`, pandas, numpy, torch (Dataset/DataLoader) |
| `src/models/` | `src.config`, `src.utils`, `src.data_alignment`, torch, scikit-learn |
| `src/xai_explainer/` | `src.config`, `src.utils`, `src.models`, shap, captum |
| `src/api/` | `src.config`, `src.utils`, `src.models`, `src.xai_explainer`, `src.data_engine` (storage), fastapi, pydantic |
| `src/dashboard/` | `src.config`, `src.utils`, streamlit, plotly, requests (API çağrıları) |

---

## 4. Yeni Dosya Ekleme Rehberi

Yeni bir dosya/modül eklerken aşağıdaki tabloyu takip et:

| Ne yapıyorsun? | Nereye ekle? | Örnek |
|---|---|---|
| Yeni bir veri kaynağı bağlayıcısı | `src/data_engine/` | `src/data_engine/sec_edgar_connector.py` |
| Yeni bir teknik gösterge | `src/feature_engineering/technical_indicators.py` | Fonksiyon olarak mevcut dosyaya ekle |
| Yeni bir NLP preprocessing adımı | `src/feature_engineering/text_preprocessor.py` | Fonksiyon olarak mevcut dosyaya ekle |
| Yeni bir embedding modeli | `src/feature_engineering/` | `src/feature_engineering/roberta_embedder.py` |
| Yeni bir model mimarisi | `src/models/` | `src/models/transformer_branch.py` |
| Yeni bir loss fonksiyonu | `src/models/loss_functions.py` | Fonksiyon/class olarak mevcut dosyaya ekle |
| Yeni bir XAI yöntemi | `src/xai_explainer/` | `src/xai_explainer/lime_explainer.py` |
| Yeni bir API endpoint'i | `src/api/routes/` | `src/api/routes/monitor.py` |
| Yeni bir dashboard sayfası | `src/dashboard/pages/` | `src/dashboard/pages/4_Data_Explorer.py` |
| Yeni bir utility fonksiyonu | `src/utils/` | Mevcut dosyaya veya `src/utils/helpers.py` |
| Yeni bir script | `scripts/` | `scripts/export_predictions.py` |
| Yeni bir test | `tests/` | `tests/test_<modül_adı>.py` |

---

## 5. `__init__.py` Kuralları

Her modülün `__init__.py` dosyasında public API'si export edilmeli:

```python
# src/data_engine/__init__.py
"""Data ingestion engine for market and sentiment data."""

from src.data_engine.alpaca_connector import AlpacaDataCollector
from src.data_engine.reddit_collector import RedditCollector
from src.data_engine.news_collector import NewsCollector
from src.data_engine.storage_manager import StorageManager
from src.data_engine.yfinance_connector import YFinanceDataCollector

__all__ = [
    "AlpacaDataCollector",
    "RedditCollector",
    "NewsCollector",
    "StorageManager",
    "YFinanceDataCollector",
]
```

---

## 6. Circular Import Önleme

Circular import riskini önlemek için:

1. **TYPE_CHECKING guard** kullan:
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.hybrid_network import MarketPulseNet
```

2. Alt katmanlar, üst katmanlardaki tiplere ihtiyaç duyduğunda **Protocol** veya **ABC** kullan
3. Fonksiyon içi (lazy) import sadece son çare olarak kullan ve nedenini yorum olarak belirt

---

## 7. Konfigürasyon Katmanı Kuralları

- Tüm hiperparametreler `config/default.yaml` ve `src/config/settings.py`'de tanımlanır
- Kod içinde magic number **yasak** (bkz: `coding_standards.md`, Bölüm 10)
- Environment-specific override'lar: `config/development.yaml`, `config/production.yaml`
- Runtime'da `get_settings()` ile erişilir (cached singleton)

---

## 8. API Route Ekleme Protokolü

Yeni bir FastAPI endpoint eklerken:

1. `src/api/schemas.py`'ye Pydantic request/response modellerini ekle
2. `src/api/routes/` altına route dosyasını oluştur
3. `src/api/main.py`'de router'ı register et
4. `docs/API_REFERENCE.md`'yi güncelle
5. `tests/test_api.py`'ye test ekle

---

## 9. Dashboard Page Ekleme Protokolü

Yeni bir Streamlit sayfası eklerken:

1. `src/dashboard/pages/` altına `<sıra>_<İsim>.py` formatında dosya oluştur
2. API endpoint'lerini `requests` ile çağır (doğrudan model import YASAK)
3. Custom component gerekiyorsa `src/dashboard/components/` altına ekle
4. Dark glassmorphic tema uyumluluğunu kontrol et

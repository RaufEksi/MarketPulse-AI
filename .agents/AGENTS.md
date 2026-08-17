# MarketPulse AI — AI IDE Kural Dosyası

> Bu dosya, AI IDE'nin (Antigravity / Gemini) projeyi anlaması ve tutarlı, kaliteli kod üretmesi için ana referans kaynağıdır.

---

## 🎯 Proje Kimliği

**MarketPulse AI**, kısa vadeli finansal volatilite sıçramalarını (≥%15 ATR artışı / 30 dakika) tahmin eden, çok modallı (multi-modal) hibrit derin öğrenme sistemidir. Fiyat yönü (yukarı/aşağı) değil, **risk ve volatilite** tahmin edilir.

Hedef kullanıcı: Portföy yöneticileri, risk algoritmaları, quant traderlar.

---

## 🛠️ Tech Stack

| Katman | Teknoloji | Notlar |
|--------|-----------|--------|
| **Dil** | Python 3.10+ | Minimum 3.10, hedef 3.11. 3.14 ile bağımlılık uyumsuzlukları var |
| **Zaman Serisi** | PyTorch (Bi-LSTM, TCN) | `src/models/time_series_branch.py` |
| **NLP** | HuggingFace FinBERT (`ProsusAI/finbert`) | 768-D CLS token embedding |
| **Füzyon** | Multi-Head Cross-Attention | Text=Query, Price=Key/Value |
| **Feature Eng.** | Pandas, NumPy, ta kütüphanesi | ATR, RSI, MACD, Bollinger |
| **XAI** | SHAP, Captum (Integrated Gradients) | `src/xai_explainer/` |
| **API** | FastAPI + Pydantic v2 | `src/api/` |
| **Dashboard** | Streamlit + Plotly | `src/dashboard/` |
| **Veri Depolama** | Parquet (PyArrow) | `data/raw/`, `data/processed/` |
| **DevOps** | Docker, docker-compose, GitHub Actions | `.github/workflows/ci.yml` |
| **Config** | Pydantic Settings + YAML | `src/config/settings.py`, `config/default.yaml` |

---

## 📁 Modül Haritası

```
src/
├── config/                 → Pydantic Settings, YAML loader
├── data_engine/            → API connector'lar (Alpaca, Yahoo, Reddit, NewsAPI)
├── feature_engineering/    → Teknik göstergeler, FinBERT embedding, labeler, text preprocessor
├── data_alignment/         → Exponential decay temporal alignment, dataset builder
├── models/                 → Bi-LSTM, TCN, CrossAttention, MarketPulseNet, Focal Loss, Trainer
├── xai_explainer/          → SHAP, Integrated Gradients, Risk Attribution
├── api/                    → FastAPI app, Pydantic schemas, route'lar
├── dashboard/              → Streamlit app, pages, components
└── utils/                  → Logger, exceptions, metrics
```

---

## 📏 Kodlama Standartları (Özet)

> Detaylı standartlar için bkz: `.agents/rules/coding_standards.md`

1. **Formatter**: Black — satır uzunluğu 100
2. **Import sıralama**: isort (profil: black) — stdlib → third-party → local
3. **Docstring**: Google style — tüm public fonksiyonlarda zorunlu
4. **Type hints**: Tüm fonksiyon parametreleri ve return değerleri için zorunlu
5. **Linter**: Flake8 (max-line-length=100), mypy (strict opsiyonel)
6. **Güvenlik**: Bandit taraması, API key hardcode yasak
7. **print() yasak**: `src/utils/logger.py`'deki structured JSON logger kullan
8. **Config erişimi**: `get_settings()` singleton fonksiyonunu kullan

---

## 📐 Mimari Kurallar (Özet)

> Detaylı kurallar için bkz: `.agents/rules/architecture_guard.md`

- Katman bağımlılık yönü: `data_engine → feature_engineering → data_alignment → models → api → dashboard`
- **Tersi yasak**: dashboard, doğrudan model veya data_engine import edemez
- Tüm konfigürasyon `src/config/settings.py` üzerinden erişilir
- Exception'lar `src/utils/exceptions.py`'deki hiyerarşiden türetilir
- Yeni bir dış API eklenirse, `src/data_engine/` altına konur

---

## 🔀 Git & Commit Kuralları

### Branch adlandırma
```
feat/<issue-number>-<kısa-açıklama>
bugfix/<issue-number>-<kısa-açıklama>
refactor/<kısa-açıklama>
docs/<kısa-açıklama>
```

### Commit mesaj formatı
```
<type>(<scope>): <subject>

<body>

Fixes #<issue-number>
```

**Tipler**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `ci`, `chore`

**Scope örnekleri**: `data-engine`, `feature-eng`, `models`, `api`, `dashboard`, `xai`, `config`, `infra`

---

## 🧪 Test Kuralları

- Her yeni public fonksiyon için en az bir unit test yazılmalı
- Test dosyası: `tests/test_<modül_adı>.py`
- Hedef coverage: **≥%80**
- Marker'lar: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.gpu`
- External API çağrıları mock'lanmalı (Alpaca, Reddit, NewsAPI)
- PyTorch model testleri küçük dummy tensor'lerle yapılmalı

---

## 🔐 Güvenlik Kuralları

- API key'ler SADECE `.env` dosyasından okunur, kod içinde hardcode **YASAK**
- `.env` dosyası **asla** commit edilmez (`.gitignore`'da)
- Log mesajlarında API key/secret yazdırılmaz
- `bandit -r src/ -ll` güvenlik taramasından geçmeli

---

## 📚 Referans Dokümanlar

- [ROADMAP & Sprint Planı](docs/ROADMAP.md) — 8 sprint'lik geliştirme takvimi
- [ISSUES Kataloğu](docs/ISSUES.md) — 33 granüler issue ve kabul kriterleri
- [Sprint Durumu](docs/SPRINT_STATUS.md) — Canlı sprint takibi
- [Mimari Karar Günlüğü](docs/DECISIONS.md) — Teknik ADR kayıtları
- [Architecture](docs/ARCHITECTURE.md) — Sistem mimarisi
- [Model Design](docs/MODEL_DESIGN.md) — MarketPulseNet detayları
- [Data Pipeline](docs/DATA_PIPELINE.md) — Veri pipeline formülleri
- [CONTRIBUTING](CONTRIBUTING.md) — PR süreci ve code review

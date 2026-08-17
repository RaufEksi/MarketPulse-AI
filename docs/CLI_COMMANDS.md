# MarketPulse AI — Terminal Komutları ve CLI Referans Rehberi 💻

Bu döküman, **MarketPulse AI** projesini çalıştırmak, eğitmek, simüle etmek, test etmek ve yönetmek için kullanabileceğiniz tüm terminal komutlarını derler.

---

## 📑 İçindekiler
1. [Web Servisleri (API & Streamlit Arayüzü)](#1-web-servisleri-api--streamlit-arayüzü)
2. [CLI Veri & Model Eğitimi Scriptleri](#2-cli-veri--model-eğitimi-scriptleri)
3. [XAI & Backtest Simülasyonu](#3-xai--backtest-simülasyonu)
4. [cURL ile API İstekleri](#4-curl-ile-api-istekleri)
5. [Test & Kod Kalitesi (Pytest & Linting)](#5-test--kod-kalitesi-pytest--linting)
6. [Docker & Kubernetes Yönetimi](#6-docker--kubernetes-yönetimi)
7. [Makefile Kısayolları](#7-makefile-kısayolları)

---

## 1. Web Servisleri (API & Streamlit Arayüzü)

### 🌐 FastAPI Backend Sunucusunu Başlatma
```bash
# Terminal 1: Geliştirici modunda (Hot Reload) başlatır
.venv/bin/uvicorn src.api.main:app --reload --port 8000
```
- **Kök Dizin (Karşılama & Linkler):** `http://localhost:8000/`
- **İnteraktif Swagger Dokümantasyonu:** `http://localhost:8000/docs`
- **ReDoc Dokümantasyonu:** `http://localhost:8000/redoc`

### 📊 Streamlit Finansal Terminal Dashboard Başlatma
```bash
# Terminal 2: Streamlit arayüzünü başlatır
.venv/bin/streamlit run src/dashboard/app.py
```
- **Dashboard Adresi:** `http://localhost:8501`

---

## 2. CLI Veri & Model Eğitimi Scriptleri

### 📥 Tarihsel Veri İndirme (Parquet Data Lake Oluşturma)
```bash
# SPY için son 14 günlük 5 dakikalık barları ve sentetik haberleri indirir
.venv/bin/python scripts/download_historical_data.py --symbol SPY --days 14

# Farklı bir hisse için örnek (Örn: AAPL, NVDA, TSLA)
.venv/bin/python scripts/download_historical_data.py --symbol NVDA --days 30
```

### 🧠 Multi-Modal PyTorch Model Eğitimi
```bash
# 5 epoch ve 32 batch boyutu ile BiLSTM + FinBERT + Cross-Attention modelini eğitir
.venv/bin/python scripts/train_model.py --epochs 5 --batch-size 32

# TCN zaman serisi kodlayıcısı ile eğitmek için
.venv/bin/python scripts/train_model.py --epochs 10 --batch-size 64 --lr 0.0005
```

### 📈 Klasik ML vs Hibrit Derin Öğrenme Benchmark Kıyaslaması
```bash
# HistGradientBoosting ve Random Forest modellerini eğitir ve metrikleri basar
.venv/bin/python scripts/evaluate_benchmark.py
```

---

## 3. XAI & Backtest Simülasyonu

### 📉 Risk Hedge Backtest Simülatörü
```bash
# Varsayılan SPY volatilite hedge simülasyonunu çalıştırır
.venv/bin/python scripts/run_backtest.py

# Özel parametrelerle backtest simülasyonu
.venv/bin/python scripts/run_backtest.py --symbol NVDA --threshold 0.70 --hedge-ratio 0.20 --capital 250000
```

---

## 4. cURL ile API İstekleri

*(FastAPI sunucusu port 8000'de açıkken yeni bir terminalden çalıştırın)*

### A. Sağlık Kontrolü (Health Check)
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

### B. Canlı Volatilite Sıçrama Tahmini (`/predict`)
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "ohlcv_bars": [
      {"timestamp": "2026-08-17T15:00:00Z", "open": 550.0, "high": 551.2, "low": 549.5, "close": 550.8, "volume": 50000},
      {"timestamp": "2026-08-17T15:05:00Z", "open": 550.8, "high": 552.0, "low": 550.2, "close": 551.5, "volume": 62000},
      {"timestamp": "2026-08-17T15:10:00Z", "open": 551.5, "high": 552.8, "low": 551.0, "close": 552.3, "volume": 75000},
      {"timestamp": "2026-08-17T15:15:00Z", "open": 552.3, "high": 553.0, "low": 551.8, "close": 552.6, "volume": 80000},
      {"timestamp": "2026-08-17T15:20:00Z", "open": 552.6, "high": 553.5, "low": 552.0, "close": 553.1, "volume": 91000},
      {"timestamp": "2026-08-17T15:25:00Z", "open": 553.1, "high": 553.8, "low": 552.5, "close": 553.4, "volume": 67000},
      {"timestamp": "2026-08-17T15:30:00Z", "open": 553.4, "high": 554.0, "low": 552.9, "close": 553.8, "volume": 88000},
      {"timestamp": "2026-08-17T15:35:00Z", "open": 553.8, "high": 554.5, "low": 553.2, "close": 554.1, "volume": 95000},
      {"timestamp": "2026-08-17T15:40:00Z", "open": 554.1, "high": 554.8, "low": 553.5, "close": 554.4, "volume": 72000},
      {"timestamp": "2026-08-17T15:45:00Z", "open": 554.4, "high": 555.0, "low": 553.8, "close": 554.7, "volume": 84000},
      {"timestamp": "2026-08-17T15:50:00Z", "open": 554.7, "high": 555.2, "low": 554.1, "close": 555.0, "volume": 90000},
      {"timestamp": "2026-08-17T15:55:00Z", "open": 555.0, "high": 555.5, "low": 554.3, "close": 555.2, "volume": 65000},
      {"timestamp": "2026-08-17T16:00:00Z", "open": 555.2, "high": 555.8, "low": 554.6, "close": 555.5, "volume": 78000},
      {"timestamp": "2026-08-17T16:05:00Z", "open": 555.5, "high": 556.0, "low": 554.9, "close": 555.8, "volume": 82000},
      {"timestamp": "2026-08-17T16:10:00Z", "open": 555.8, "high": 556.3, "low": 555.1, "close": 556.0, "volume": 71000},
      {"timestamp": "2026-08-17T16:15:00Z", "open": 556.0, "high": 556.5, "low": 555.3, "close": 556.2, "volume": 89000},
      {"timestamp": "2026-08-17T16:20:00Z", "open": 556.2, "high": 556.7, "low": 555.5, "close": 556.4, "volume": 93000},
      {"timestamp": "2026-08-17T16:25:00Z", "open": 556.4, "high": 557.0, "low": 555.7, "close": 556.6, "volume": 86000},
      {"timestamp": "2026-08-17T16:30:00Z", "open": 556.6, "high": 557.2, "low": 556.0, "close": 556.8, "volume": 99000},
      {"timestamp": "2026-08-17T16:35:00Z", "open": 556.8, "high": 557.5, "low": 556.2, "close": 557.1, "volume": 110000}
    ]
  }' | python -m json.tool
```

### C. Açıklanabilir Yapay Zeka SHAP Analizi (`/explain`)
```bash
curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "mp-live-001", "symbol": "NVDA", "top_k_features": 4}' | python -m json.tool
```

### D. Algoritmik Backtest Simülasyonu (`/backtest`)
```bash
curl -s -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "spike_threshold": 0.65, "hedge_reduction_factor": 0.2, "initial_capital": 100000.0}' | python -m json.tool
```

### E. Prometheus Metrikleri (`/metrics`)
```bash
curl -s http://localhost:8000/metrics
```

---

## 5. Test & Kod Kalitesi (Pytest & Linting)

### 🧪 Testleri Çalıştırma & Kapsam Raporu
```bash
# Tüm testleri detaylı çalıştırır ve terminalde dosya dosya kapsamı gösterir
.venv/bin/pytest -v --cov=src --cov-report=term-missing

# Test sonucunda HTML raporu oluşturmak için (htmlcov/index.html)
.venv/bin/pytest --cov=src --cov-report=html
```

### 🔍 Kod Formatı ve Statik Analiz
```bash
# Otomatik kod formatlama (Black & isort)
.venv/bin/black src/ tests/ --line-length=100
.venv/bin/isort src/ tests/

# Stil ve Linter kontrolü
.venv/bin/flake8 src/ tests/ --max-line-length=100

# Statik Tip Kontrolü (mypy)
.venv/bin/mypy src/ --ignore-missing-imports

# Güvenlik Açığı Taraması (bandit)
.venv/bin/bandit -r src/ -ll
```

---

## 6. Docker & Kubernetes Yönetimi

### 🐳 Docker & Docker Compose
```bash
# Docker imajlarını derler
docker-compose build

# API ve Dashboard konteynerlerini arka planda başlatır
docker-compose up -d

# Logları canlı takip eder
docker-compose logs -f

# Servisleri durdurur
docker-compose down
```

### ☸️ Kubernetes Dağıtımı
```bash
# Konfigürasyon, Deployment, HPA ve Ingress'leri Kubernetes kümesine uygular
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/deployment-api.yaml
kubectl apply -f deploy/kubernetes/deployment-dashboard.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/hpa.yaml
kubectl apply -f deploy/kubernetes/ingress.yaml

# Pod durumlarını inceler
kubectl get pods -l app=marketpulse-api
```

---

## 7. Makefile Kısayolları

Proje kök dizininde `make` kullanarak sık yapılan görevleri hızlıca tetikleyebilirsiniz:

```bash
make help          # Tüm kullanılabilir make komutlarını listeler
make test          # Pytest ve coverage çalıştırır
make format        # Black ve isort ile kodları formatlar
make lint          # Flake8, mypy ve bandit denetimini yapar
make train         # Model eğitim scriptini başlatır
make api           # FastAPI sunucusunu ayağa kaldırır
make dashboard     # Streamlit arayüzünü ayağa kaldırır
make docker-up     # Docker konteynerlerini başlatır
make clean         # __pycache__, geçici dosyalar ve test artıklarını temizler
```

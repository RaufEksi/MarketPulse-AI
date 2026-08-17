# MarketPulse AI — Deployment Guide

This guide covers local development, Docker containerization, Kubernetes orchestration, and monitoring setup for **MarketPulse AI**.

---

## 1. Local Development

```bash
# 1. Clone repository
git clone https://github.com/RaufEksi/MarketPulse-AI.git
cd MarketPulse-AI

# 2. Configure environment
cp .env.example .env
# Fill in ALPACA_API_KEY, REDDIT_CLIENT_ID, etc.

# 3. Setup virtual environment & dependencies
make setup

# 4. Run tests
make test

# 5. Launch FastAPI backend & Streamlit Dashboard
make api        # Terminal 1: http://localhost:8000
make dashboard  # Terminal 2: http://localhost:8501
```

---

## 2. Docker & Docker Compose

```bash
# Build and run the complete multi-container stack
docker-compose up -d --build

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop services
docker-compose down
```

Services exposed:
- **FastAPI Backend**: `http://localhost:8000` (Docs: `/docs`)
- **Streamlit Terminal**: `http://localhost:8501`
- **Prometheus Metrics**: `http://localhost:9090`

---

## 3. Kubernetes Deployment

```bash
kubectl apply -f deploy/kubernetes/deployment-api.yaml
kubectl apply -f deploy/kubernetes/deployment-dashboard.yaml
kubectl apply -f deploy/kubernetes/service.yaml
```

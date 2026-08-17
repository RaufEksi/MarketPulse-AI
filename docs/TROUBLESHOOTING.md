# MarketPulse AI — Troubleshooting & FAQ

This document provides diagnostic solutions for common configuration, data ingestion, GPU acceleration, and inference challenges.

---

## 1. CUDA & PyTorch GPU Allocation

**Issue:** `RuntimeError: CUDA out of memory` during FinBERT embedding or training.  
**Resolution:**
1. Reduce `nlp.batch_size` from `32` to `16` or `8` in `config/default.yaml`.
2. Enable PyTorch gradient checkpointing or mixed-precision training (`torch.cuda.amp.autocast()`).
3. If running on CPU, set `nlp.device: "cpu"` and `model.device: "cpu"`.

---

## 2. Alpaca API & Rate Limits

**Issue:** `HTTP 429 Too Many Requests` or `Invalid API Key`.  
**Resolution:**
1. Verify `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`.
2. Check if using Paper Trading endpoint (`https://paper-api.alpaca.markets`) or Live endpoint.
3. MarketPulse AI automatically backs off with exponential jitter when HTTP 429 occurs.

---

## 3. Class Imbalance & Model Metric Optimization

**Issue:** Model predicts all 0s (no volatility spikes) due to extreme class imbalance.  
**Resolution:**
1. Verify `loss_type: "focal"` in `config/default.yaml` with `focal_gamma: 2.0` and `focal_alpha: 0.75`.
2. Evaluate models using **PR-AUC** (Precision-Recall AUC) instead of standard accuracy or ROC-AUC.

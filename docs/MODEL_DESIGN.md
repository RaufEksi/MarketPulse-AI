# MarketPulse AI - Model Architecture & Design 🧠

This document details the hybrid deep learning architecture, attention fusion mechanisms, and loss functions powering **MarketPulse AI**.

---

## 1. Network Overview (`MarketPulseNet`)

```
   Time-Series Input [B, 78, 16]          Text Embedding [B, 768]
               │                                      │
       ┌───────┴────────┐                     ┌───────┴────────┐
       │  Bi-LSTM / TCN │                     │ Dense & LN     │
       │   Encoder      │                     │ Projection     │
       └───────┬────────┘                     └───────┬────────┘
               │                                      │
      h_ts: [B, 78, 128]                     h_text: [B, 1, 128]
               │                                      │
               └───────────────┬──────────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │ Multi-Head Cross-Attention     │
               │ Query: Text, Key/Value: Price  │
               └───────────────┬────────────────┘
                               │
                      Fusion Representation
                               │
               ┌───────────────▼────────────────┐
               │ Dense (128 -> 64 -> Dropout)   │
               │ Classification Head (Linear)   │
               └───────────────┬────────────────┘
                               │
               Volatility Spike Probability (0 to 1)
```

---

## 2. Component Specifications

### 2.1 Time-Series Branch (`TimeSeriesEncoder`)
- **Inputs**: Shape `[Batch, Sequence_Length=78, Input_Dim=16]`.
- **Architecture**:
  - 2-Layer Bidirectional LSTM (`hidden_dim=64` per direction $\rightarrow 128$).
  - Optional: Causal Dilated Temporal Convolutional Network (TCN) with residual blocks.
  - Layer Normalization & Dropout ($p=0.2$).

### 2.2 NLP Text Branch (`TextProjectionEncoder`)
- **Inputs**: Aligned 768-D FinBERT embeddings.
- **Architecture**:
  - `Linear(768, 256)` $\rightarrow$ `LayerNorm` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(256, 128)`.

### 2.3 Multi-Head Cross-Attention Fusion (`CrossAttentionFusion`)
- Fuses text and price context using scaled dot-product cross-attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- Allows breaking news embeddings ($Q$) to attend to specific historical price bar sequences ($K, V$).

---

## 3. Loss Function: Focal Loss

To address severe class imbalance in market volatility spikes ($\approx 5-10\%$ positive instances), MarketPulseNet uses **Binary Focal Loss**:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- Focusing parameter: $\gamma = 2.0$ (downweights easy normal market instances).
- Balancing parameter: $\alpha = 0.75$ (penalizes false negatives on volatility spikes).

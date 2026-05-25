# Architecture & Training Comparison
**how2sign (3).ipynb · notebookc5efacafaa (1).ipynb · Proposed Improved Version**

---

## Key Finding

> Both `how2sign (3)` and `notebookc5efacafaa (1)` share an **identical architecture**.  
> The difference is that `notebookc5efacafaa` loaded pre-trained weights and continued training — yet still reached WER 0.9992.  
> The core problems are architectural and pre-processing, not just training length.

### Headline Numbers

| Metric | Value |
|---|---|
| Architecture relationship | Identical (same layers, same units, same hyperparams) |
| notebookc5efacafaa WER | **0.9992 (99.9% wrong)** |
| Best val_loss achieved | 85.09 (epoch 9) |
| Epochs before early stop | 9 out of 30 |

---

## Why Same Architecture, Different Results?

| | how2sign (3).ipynb | notebookc5efacafaa (1).ipynb |
|---|---|---|
| Weight initialisation | Random (fresh start) | Loaded from pre-trained checkpoint |
| Epoch 1 loss | ~2118 (expected for fresh CTC) | ~96.3 (head start from weights) |
| Epochs trained | Designed for 30 | Stopped at 9 (early stopping) |
| WER | Not evaluated | **0.9992** |
| Val loss spikes | Unknown | Yes — epoch 3: 303, epoch 7: 109 |

---

## Layer-by-Layer Comparison

| Layer | how2sign (3) & notebookc5e (identical) | Proposed | Change |
|---|---|---|---|
| Input projection | – | Dense(128) + LayerNorm | NEW |
| Temporal downsampling | – | Conv1D(128, kernel=3, stride=2) | NEW |
| BiLSTM 1 | BiLSTM(256×2 = 512) | BiLSTM(256×2 = 512) | same |
| Norm after BiLSTM 1 | **BatchNorm** | LayerNorm | FIXED |
| Attention 1 | MHA(4 heads, key_dim=64) | MHA(4 heads, key_dim=64) | same |
| Residual + Norm | Add + LayerNorm | Add + LayerNorm | same |
| Dropout 1 | **0.2** | 0.35 | FIXED |
| BiLSTM 2 | **BiLSTM(256×2 = 512) [1.57M!]** | BiLSTM(128×2 = 256) | FIXED |
| Norm after BiLSTM 2 | **BatchNorm** | LayerNorm | FIXED |
| Attention 2 | – | MHA(4 heads, key_dim=32) | NEW |
| Dropout 2 | **0.2** | 0.35 | FIXED |
| Bottleneck Dense | – | Dense(256, relu) | NEW |
| Dropout 3 | – | 0.2 | NEW |
| Output logits | Dense(501) | Dense(501) | same |
| Output activation | Softmax → CTC | Softmax → CTC | same |

---

## Parameter Count

### Original (both notebooks — identical)

| Layer | Params |
|---|---|
| BiLSTM 1 (256 units) | 1,001,472 |
| BatchNorm | 2,048 |
| MHA (4 heads, key_dim=64) | 525,568 |
| LayerNorm | 1,024 |
| BiLSTM 2 (256 units) — **oversized** | **1,574,912** |
| BatchNorm | 2,048 |
| Dense(501) | 257,013 |
| **Total** | **3,364,085** |

### Proposed Architecture

| Layer | Params |
|---|---|
| Dense(128) projection + LayerNorm | ~30K |
| Conv1D(128, stride=2) | ~50K |
| BiLSTM 1 (256 units) | ~530K |
| LayerNorm + MHA 1 | ~527K |
| BiLSTM 2 (128 units) — halved | ~430K |
| LayerNorm + MHA 2 | ~135K |
| Dense(256) bottleneck | ~66K |
| Dense(501) | ~129K |
| **Total** | **~1.9M (44% smaller)** |

---

## Training Configuration Differences

| Setting | how2sign (3) & notebookc5e | Proposed |
|---|---|---|
| Feature normalization | **None** (raw coords, mean abs ~200) | Per-sample zero-mean, unit-std |
| L2 regularization | **1e-4** (too weak) | 5e-4 |
| Learning rate | 1e-4 | 5e-4 |
| Gradient clipnorm | **5.0** | 1.0 (tighter) |
| EarlyStopping patience | **5** (too low for noisy CTC) | 8 |
| EarlyStopping min_delta | **1.0** (too coarse) | 0.5 |
| ReduceLROnPlateau patience | 3 | 4 |
| Decoding at evaluation | **Greedy** | Beam search (width=10) |
| Vocabulary size | **500 tokens** | 2000 tokens (fewer UNK) |

---

## Actual vs Projected Results

| Metric | how2sign (3) | notebookc5efacafaa (1) | Proposed (projected) |
|---|---|---|---|
| Epoch 1 loss | ~2118 (from scratch) | 96.3 (loaded weights) | ~40–60 (normalized input) |
| Best val_loss | Not reported | 85.09 (epoch 9) | ~30–45 |
| Epochs trained | 30 (or early stop) | **9 (early stop)** | 20–30 (looser callbacks) |
| WER (test) | Not evaluated | **0.9992 (99.9% wrong)** | ~0.6–0.8 (projected) |
| Val loss spikes | Unknown | Yes — epoch 3: 303, epoch 7: 109 | Eliminated (LayerNorm) |

---

## Root Causes of WER 0.9992

### #1 — No feature normalization (biggest impact)
Raw OpenPose coordinates have mean absolute values of **141–249**. LSTMs expect inputs near zero. This is the primary cause of slow convergence.

**Fix:** Normalize each sample per-feature (zero mean, unit std) inside the data generator.

---

### #2 — BatchNorm on padded sequences
`BatchNormalization` normalizes across batch + time dimensions together. Padded zeros drag the mean down and cause the val_loss spikes to **303 at epoch 3** and **109 at epoch 7**.

**Fix:** Replace all `BatchNormalization` with `LayerNormalization`.

---

### #3 — Early stopping too aggressive
`min_delta=1.0` with `patience=5` on noisy CTC loss cuts training at epoch 9. The model has not converged — CTC loss needs to reach ~20–30 for meaningful decoding.

**Fix:** `patience=8`, `min_delta=0.5`.

---

### #4 — No input projection layer
Raw 232-dim coordinates feed directly into the BiLSTM with no embedding step. The LSTM must simultaneously learn to compress features AND model sequences — two different tasks in one layer.

**Fix:** Add `Dense(128, relu)` + `LayerNorm` before the first BiLSTM.

---

### #5 — Oversized second BiLSTM
BiLSTM_2 has **1.57M params** (larger than BiLSTM_1 at 1M) — this is backwards for a 6K-sample dataset. It overfits heavily and causes poor generalisation.

**Fix:** Reduce BiLSTM_2 to 128 units per direction (~430K params).

---

### #6 — Vocabulary too small (500 tokens)
With only 500 words, many sentence tokens map to `[UNK]`. CTC can never learn to predict `[UNK]` correctly, making WER structurally high regardless of training quality.

**Fix:** Increase `MAX_TOKENS` to 2000.

---

*Source: notebookc5efacafaa (1).ipynb · how2sign (3).ipynb · May 22, 2026*

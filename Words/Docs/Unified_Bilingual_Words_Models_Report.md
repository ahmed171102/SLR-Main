# 📋 Bilingual Sign Language Words Recognition — Unified Systems Report

> **Date:** April 15, 2026  
> **Module:** `SLR Main / Words` (ASL & ArSL)  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Acquisition & Processing Pipelines](#2-data-acquisition--processing-pipelines)
   - [2.1 ASL (American Sign Language)](#21-asl-american-sign-language)
   - [2.2 ArSL (Arabic Sign Language)](#22-arsl-arabic-sign-language)
3. [Model Architectures](#3-model-architectures)
4. [Live Inference System & UX Features](#4-live-inference-system--ux-features)
5. [Comparative System Overview](#5-comparative-system-overview)
6. [Strategic Roadmap & Next Steps](#6-strategic-roadmap--next-steps)

---

## 1. Executive Summary

This comprehensive report details the dual-model architecture deployed for bilingual word-level sign language recognition. The system is split into two robust parallel pipelines: **ASL (American Sign Language)** and **ArSL (Arabic Sign Language)**. 

Both systems rely on advanced spatio-temporal modeling via **Bidirectional LSTMs (BiLSTM)** and consume a highly dense **258-feature spatial matrix** (Pose + Left Hand + Right Hand) per frame. While the ASL model relied on a custom extraction pipeline from raw videos, the ArSL model was heavily optimized to ingest pre-computed `.npy` coordinate data, cutting processing overhead by over 90%.

Both pipelines feed into a singular, highly flexible **Live Inference Application** featuring custom gestural controls ("X-Arm" deletion) and real-time scaling.

---

## 2. Data Acquisition & Processing Pipelines

### 2.1 ASL (American Sign Language)
- **Dataset:** WLASL (Word-Level American Sign Language)
- **Vocabulary:** 157 target words spanning 9 distinct categories (Verbs, Family, Adjectives, Objects, etc.)
- **Pipeline Methodology:** 
  - Raw `.mp4` video clips processed via `cv2` and `MediaPipe Holistic`.
  - Videos are uniformly sampled or zero-padded to a strict **30-frame** time series.
  - Generates `(30, 258)` tensors utilizing 132 body pose landmarks and 63 landmarks per hand.
- **Processing Time:** ~1 hour 42 minutes for 1,077 usable samples (due to heavy CPU/MediaPipe overhead).

### 2.2 ArSL (Arabic Sign Language)
- **Dataset:** KArSL-502 (King Saud University Arabic Sign Language Database)
- **Vocabulary:** Targets up to 502 words.
- **Pipeline Methodology:** 
  - **Optimization Breakthrough:** To bypass the crushing bottleneck of rendering raw video through MediaPipe, this pipeline was redesigned to scrape **pre-extracted `.npy` arrays** provided by the dataset creators directly. 
  - Keypoints are artificially padded/sampled to a strict **48-frame** sequence length.
  - Automatically merges Left and Right hand coordinate tensors.
- **Processing Time:** Under 5 minutes total for dataset caching (Zero Image-Processing required during build phase).

---

## 3. Model Architectures

Both networks are sequential Recurrent Neural Networks designed to capture the temporal evolution of hand movements alongside spatial positioning relative to the body center.

### ASL BiLSTM (30-Frame Sequence)
```text
INPUT (30, 258) → BiLSTM(128) + BN + Drop(0.3) 
                → LSTM(64) + BN + Drop(0.3) 
                → Dense(128) + ReLU + Drop(0.2) 
                → OUTPUT Softmax (157 Classes)
Total Params: ~508k
```
- **Training Results:** ~52% Training Accuracy vs ~9.8% Validation Accuracy.
- **Status:** Heavily overfit due to low samples-per-class (~6.8 videos per word). Requires aggressive data augmentation.

### ArSL BiLSTM (48-Frame Sequence)
```text
INPUT (48, 258) → BiLSTM(256) + BN + Drop(0.4) 
                → BiLSTM(128) + BN + Drop(0.4)
                → LSTM(64) + BN + Drop(0.4)
                → Dense(256) + ReLU + Drop(0.3)
                → OUTPUT Softmax (502 Classes)
```
- **Updates:** Dramatically deeper than the ASL pipeline to compensate for the massive 502-class challenge. Employs `sklearn` balanced class weights to fight representation bias. Trained strictly via Kaggle T4 x2 instances.

---

## 4. Live Inference System & UX Features

The production webcam endpoint (`ASL_Word_Live_Test` / `ArSL_Word_Live_Test`) dynamically swaps models and vocabulary mappings depending on the `.json` config cache.

| Feature | Description |
| :--- | :--- |
| **Holistic Synchronization** | Translates the live 3D webcam feed via `MediaPipe Holistic` directly into the `(Frames, 258)` required tensor representation. |
| **Z-Score Normalization** | Real-time inputs are normalized using `<lang>_scaler_stats.npz`, ensuring camera coordinates align exactly with training distributions. |
| **Multi-Modal Lexicon UI** | Real-time dual display utilizing `shared_word_vocabulary.csv` to map predicted class indices to both English and Arabic translations simultaneously. |
| **"X-Arm" Fallback Geofence** | **Physical Undo Button:** If poor tracking registers an incorrect word, the user can cross their wrists dynamically for `2 seconds` to pop the last recognized word from the UI string queue. |
| **Global Keyboard Hooks** | Manual typing controls directly into the frame window (`Q` to quit, `Backspace/D/X` to delete word, `Space` to space translation). |

---

## 5. Comparative System Overview

| Metric / Parameter | ASL Pipeline | ArSL Pipeline | 
| :--- | :--- | :--- |
| **Base Dataset** | WLASL | KArSL-502 |
| **Vocabulary Size** | 157 Active Words | 502 Active Words |
| **Sequence Length** | 30 Frames | 48 Frames |
| **Input Feature Dim** | 258 | 258 |
| **Data Scraping Tool** | MediaPipe OpenCV Loop | Direct `.npy` load mapping |
| **Extract Duration** | > 100 Minutes | < 5 Minutes |
| **Primary Optimizer** | Adam (ReduceLR via validation) | Adam (Balanced static weights) |
| **Deployment Mode** | Local GPU / CPU Forced Mode | Kaggle T4 (Train) → Local (Inference)|

---

## 6. Strategic Roadmap & Next Steps

1. **ASL Generalization Boost:**
   - The ASL model has saturated its limited training data. We must introduce an aggressive `keras` data augmentation layer mapping standard spatial jitter, temporal warping (random sequence acceleration), and horizontal/vertical flips to combat overfitting.
2. **Keypoint Scale Normalization:**
   - The KArSL dataset's pre-extracted `.npy` points may not exactly match the bounding box parameters generated by the local MediaPipe live instance. We need a diagnostic script to compare the `(Min, Max)` spatial variance of Kaggle's files vs our live webcam feed to prevent predictive drift.
3. **Cross-Pollination Feature Migration:**
   - Apply the fast `.npy` structural extraction method used in ArSL to the ASL dataset if external pre-computed landmarks become publicly available, standardizing both architectures permanently.

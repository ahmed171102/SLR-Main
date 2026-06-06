# Sign Language Recognition (SLR) System

## Project Overview

This repository contains the complete codebase and documentation for a **Sign Language Recognition (SLR)** system capable of recognizing both **American Sign Language (ASL)** and **Arabic Sign Language (ArSL)** in real-time.

The project leverages state-of-the-art machine learning models, including **MediaPipe** for pose estimation and **Transfer Learning** with MobileNetV2, to achieve real-time recognition of letters and words across multiple languages.

### Key Features
- **Real-time Letter Recognition**: Using MediaPipe landmarks and MLP classifiers.
- **Word Recognition**: Temporal sequence modeling with BiLSTM and custom attention mechanisms.
- **Multi-Language Support**: English (ASL) and Arabic (ArSL).
- **Deployment Ready**: Detailed guides for Backend (FastAPI), Web Frontend (React), and Mobile (React Native).

## 📂 Repository Structure

The repository is organized as follows:

- **`Letters/`**:
  - `ASL Letter (English)/`: Notebooks and models for English letter recognition.
  - `ArSL Letter (Arabic)/`: Notebooks and models for Arabic letter recognition.
  - `Guides/`: Detailed guides for model training and optimization.

- **`Words/`**:
  - `ASL Word (English)/`: Word-level recognition models (BiLSTM).
  - `ArSL Word (Arabic)/`: Word-level recognition models for Arabic sign language.
  - `Shared/`: Vocabulary and index mappings.

- **`Deployment/`**:
  - `docs/`: **Comprehensive Deployment Guides**, including backend API specs, frontend architecture, and cloud deployment plans.

- **`backend/`** *(In Progress)*: Python (FastAPI) backend for serving predictions.
- **`web/`** *(In Progress)*: React-based web interface.
- **`mobile/`** *(In Progress)*: React Native mobile application.
- **`scripts/`**: Utility scripts for data processing and model conversion.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ahmed171102/SLR-Main.git
    cd SLR-Main
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebooks

Navigate to `Letters/Original Notebooks` or specific language folders to run Jupyter notebooks:

```bash
jupyter notebook
```

**Key Notebooks:**
- `Letters/ASL Letter (English)/Combined_Architecture.ipynb`: Real-time ASL letter testing.
- `Letters/ASL Letter (English)/Mediapipe_Training.ipynb`: MLP training pipeline.
- `Words/ArSL Word (Arabic)/ArSL_Word_Training_CustomWords.ipynb`: Arabic word recognition (131 classes, graduation project).

## 🛠️ Deployment Plan

The project is following a 5-Phase Deployment Strategy (see `Deployment/docs/FULL_DEPLOYMENT_PLAN.md`):

1.  **Scripts**: Model conversion and data export.
2.  **Backend API**: FastAPI service with WebSocket support.
3.  **Web Frontend**: React app with real-time camera integration.
4.  **Mobile App**: React Native app using TFLite models.
5.  **Cloud Deployment**: Dockerized containers on Railway/Vercel.

## 🤝 Contributing

Contributions are welcome! Please follow the `Deployment/docs/01_INVENTORY.md` to see what is currently pending implementation.

## 📄 License

[MIT License](LICENSE)

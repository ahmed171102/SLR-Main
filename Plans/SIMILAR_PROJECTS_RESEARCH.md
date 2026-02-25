# 🔍 Similar Projects Research & Recommendations

> **Date:** 2026-02-25 12:38:53  
> **Project:** Sign Language Recognition (SLR) System  
> **Repo:** ahmed171102/SLR-Main  
> **Purpose:** Document similar open-source projects for English (ASL) & Arabic (ArSL) sign language recognition, and provide clear guidance on what to use and what NOT to use.

---

## 📋 Table of Contents

- [Your Project Summary](#your-project-summary)
- [Similar Projects Found](#similar-projects-found)
- [Detailed Comparison Table](#detailed-comparison-table)
- [✅ What to USE](#-what-to-use)
- [❌ What NOT to Use](#-what-not-to-use)
- [Datasets — Use vs. Skip](#datasets--use-vs-skip)
- [Step-by-Step Action Plan](#step-by-step-action-plan)
- [Useful Links & Resources](#useful-links--resources)

---

## Your Project Summary

Your project (`SLR-Main`) is a **Sign Language Recognition system** that supports:

| Feature | Details |
|---|---|
| **Languages** | American Sign Language (ASL) + Arabic Sign Language (ArSL) |
| **Letter Recognition** | MediaPipe landmarks + MLP classifiers |
| **Word Recognition** | BiLSTM + Custom Attention mechanism |
| **Transfer Learning** | MobileNetV2 |
| **Deployment** | FastAPI backend + React web + React Native mobile (in progress) |

---

## Similar Projects Found

### 1. ⭐ mohamedelsharkawy — English-Arabic Sign Language Recognition (BEST MATCH)

**Repo:** [mohamedelsharkawy-coder/English-Arabic-Sign-Language-Recognition-Project](https://github.com/mohamedelsharkawy-coder/English-Arabic-Sign-Language-Recognition-Project)

- **Languages:** Both English (ASL) AND Arabic (ArSL) — same as yours
- **Scope:** ~90 words per language (letters + words)
- **Approach:** MediaPipe landmarks → Random Forest Classifier
- **Dataset:** Custom webcam-collected, 360,000 frames (2,000 frames × 90 classes × 2 languages)
- **Accuracy:** 99.8% on test data
- **Deployment:** Flask REST API + ngrok + working mobile app
- **Extra Features:** Video-to-Text AND Text-to-Video conversion
- **Dataset shared?** ✅ Yes, via Google Drive (raw frames, pickle data, trained models)

### 2. abdelrhmanmousa — Arabic Sign Language Recognition

**Repo:** [abdelrhmanmousa/Arabic-Sign-Language-recognition](https://github.com/abdelrhmanmousa/Arabic-Sign-Language-recognition)

- **Languages:** Arabic only
- **Approach:**
  - **MLP classifier** for static/character-level signs (same as you!)
  - **LSTM network** for dynamic/word-level signs
- **Pipeline:** MediaPipe pose estimation → landmark extraction → deep learning classification
- **Deployment:** Local only (no API, no mobile app)
- **Dataset shared?** ❌ Not publicly shared

### 3. mahmoudmhashem — ArSLr (Arabic Sign Language Recognition)

**Repo:** [mahmoudmhashem/ArSLr](https://github.com/mahmoudmhashem/ArSLr)

- **Languages:** Arabic only
- **Two Phases:**
  - **Alphabets:** MediaPipe + SVM → 95% accuracy (32 Arabic letters)
  - **Dynamic Words:** MediaPipe + GRU → 97% accuracy
- **Dataset:** Emarat dataset (alphabets) + Mansoura University dataset (words)
- **Deployment:** Google Colab notebooks (no API, no mobile)
- **Dataset shared?** ⚠️ Partially (via Colab notebooks)

### 4. AhmedDesouki — Arabic SLR using Deep Learning

**Repo:** [AhmedDesouki/Arabic-Sign-Language-Recognition-using-Deep-Learning-Approaches](https://github.com/AhmedDesouki/Arabic-Sign-Language-Recognition-using-Deep-Learning-Approaches)

- **Languages:** Arabic only
- **Approach:** LSTM + hybrid deep learning models
- **Focus:** Complex gestures
- **Deployment:** Local only
- **Dataset shared?** ❌ Not publicly shared

---

## Detailed Comparison Table

| Feature | **Your Project (SLR-Main)** | **mohamedelsharkawy** | **abdelrhmanmousa** | **mahmoudmhashem (ArSLr)** |
|---|---|---|---|---|
| **English + Arabic** | ✅ Both | ✅ Both | ❌ Arabic only | ❌ Arabic only |
| **Letter Recognition** | MediaPipe + MLP | MediaPipe + Random Forest | MediaPipe + MLP | MediaPipe + SVM |
| **Word Recognition** | BiLSTM + Attention | Random Forest (frame-based) | LSTM | GRU |
| **Transfer Learning** | MobileNetV2 | ❌ None | ❌ None | ❌ None |
| **Dataset Size** | Custom | 360K frames (90 classes) | Custom | Emarat + Mansoura |
| **API Backend** | FastAPI (in progress) | Flask (done) | ❌ None | ❌ None |
| **Mobile App** | React Native (planned) | ✅ Working mobile app | ❌ None | ❌ None |
| **Web Frontend** | React (planned) | ❌ None | ❌ None | ❌ None |
| **Real-time** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes (Colab) |
| **Reported Accuracy** | Not listed yet | 99.8% | Not listed | 95%-97% |

---

## ✅ What to USE

### 1. USE: mohamedelsharkawy's Dataset (High Priority)
**Why:** It is the ONLY bilingual (English + Arabic) publicly available custom word-level dataset from these projects.
- 90 word classes per language
- 2,000 frames per class
- Already extracted as MediaPipe landmarks in pickle format
- **Download links:**
  - [English frames](https://drive.google.com/drive/folders/1IBbyj8CzZjeDakChW7XoDqIErlL4ycQX?usp=sharing)
  - [Arabic frames](https://drive.google.com/drive/folders/1qEWWjAK1TAc4zRbwk7aMqvyu5PwZqOIr?usp=sharing)
  - [Pre-extracted landmarks (pickle)](https://drive.google.com/drive/folders/1KavCk5ktwkwO3dUtO03pRv-4dJTWb6ly?usp=sharing)
  - [Trained models (pickle)](https://drive.google.com/drive/folders/1kVKa9UZAuisZ8y44m59GAkmCdqI3UmFs?usp=sharing)

### 2. USE: Your BiLSTM + Attention Architecture (Keep It!)
**Why:** Your architecture is the MOST advanced of all these projects.
- BiLSTM captures temporal sequences better than Random Forest or basic LSTM
- Attention mechanism highlights the most important frames
- None of the other projects use attention — this is your competitive advantage
- **Action:** Train your BiLSTM on mohamedelsharkawy's dataset to potentially beat their 99.8%

### 3. USE: Your MobileNetV2 Transfer Learning (Keep It!)
**Why:** No other project uses transfer learning. This gives you:
- Better feature extraction from raw images
- Works well with smaller datasets
- Industry-standard approach for mobile deployment

### 4. USE: Your FastAPI Backend (Keep It, Don't Switch to Flask!)
**Why:** FastAPI is superior to Flask:
- Async support (better for WebSocket real-time video)
- Auto-generated API docs (Swagger/OpenAPI)
- Better performance
- **Action:** Study mohamedelsharkawy's Flask API patterns for endpoint design, then implement in FastAPI

### 5. USE: ArSLr's GRU Approach as a Baseline Comparison
**Why:** GRU is simpler than BiLSTM and faster to train.
- Try it as a secondary model to compare speed vs. accuracy
- Their 97% accuracy on Arabic words is a good benchmark
- **Don't replace** your BiLSTM — just use GRU as a comparison experiment

### 6. USE: Public Arabic Datasets to Augment Training Data
- **KArSL Dataset** — 502 Arabic sign words (largest Arabic SL dataset): [https://hamzah-luqman.github.io/KArSL/](https://hamzah-luqman.github.io/KArSL/)
- **ArabicSL-Net** — 307 Arabic words, ~30,000 videos (Creative Commons license): [https://zenodo.org/records/7771372](https://zenodo.org/records/7771372)

### 7. USE: abdelrhmanmousa's MLP + LSTM Architecture Patterns
**Why:** Their architecture is nearly identical to yours (MLP for letters + LSTM for words).
- Good for validating your approach
- Compare their LSTM word recognition with your BiLSTM
- Useful for debugging if you hit accuracy issues

---
## ❌ What NOT to Use

### 1. DON'T USE: Random Forest for Word Recognition
**Why:** mohamedelsharkawy uses Random Forest for word-level classification, which:
- Treats each frame independently (no temporal understanding)
- Cannot capture motion/sequence patterns in sign language
- Your BiLSTM is fundamentally better for sequential word gestures
- **Their high accuracy (99.8%) is likely on a controlled dataset** — it may not generalize well

### 2. DON'T USE: SVM for Letter Classification
**Why:** ArSLr uses SVM which achieves only 95% accuracy.
- Your MLP approach is more flexible and scalable
- MLPs can learn non-linear decision boundaries better
- SVM struggles with high-dimensional landmark data at scale

### 3. DON'T USE: Flask for Your Backend
**Why:** You already have FastAPI planned, which is better in every way:
- Don't downgrade to Flask just because mohamedelsharkawy used it
- FastAPI has native WebSocket support (critical for real-time video)
- FastAPI is faster and more modern

### 4. DON'T USE: ngrok for Production Deployment
**Why:** mohamedelsharkawy uses ngrok which is:
- Temporary tunnels only (not production-ready)
- Unreliable for real users
- **Instead:** Stick with your plan: Docker → Railway/Vercel for cloud deployment

### 5. DON'T USE: Google Colab as Your Primary Platform
**Why:** ArSLr runs entirely in Colab which:
- Has session timeouts
- Limited GPU hours
- Not suitable for deployment
- **Your approach** (local Jupyter → backend API → web/mobile) is far more professional

### 6. DON'T USE: Pickle for Model Serialization in Production
**Why:** mohamedelsharkawy saves models as `.pickle` files which:
- Are not secure (pickle can execute arbitrary code)
- Not portable across Python versions
- **Instead:** Keep using `.h5` (Keras) and `.tflite` (TensorFlow Lite) formats as you already plan

---
## Datasets — Use vs. Skip

| Dataset | Source | Use? | Reason |
|---|---|---|---|
| **mohamedelsharkawy's custom dataset** | Google Drive | ✅ **YES** | Only bilingual English+Arabic word-level dataset; matches your scope exactly |
| **KArSL (502 Arabic words)** | [KArSL](https://hamzah-luqman.github.io/KArSL/) | ✅ **YES** | Largest Arabic SL dataset; great for Arabic word augmentation |
| **ArabicSL-Net (307 words, 30K videos)** | [Zenodo](https://zenodo.org/records/7771372) | ✅ **YES** | Open license (CC 4.0); good variety of Arabic signs |
| **ArSLr's Emarat dataset** | Via Colab | ⚠️ **MAYBE** | Arabic letters only; you may already have enough letter data |
| **ArSLr's Mansoura University dataset** | Via Colab | ⚠️ **MAYBE** | Arabic words; access may be limited |
| **WLASL (ASL words)** | [WLASL GitHub](https://github.com/dxli94/WLASL) | ✅ **YES** | Large-scale ASL word dataset; great for English augmentation |

---
## Step-by-Step Action Plan

### Phase 1: Data Augmentation (Week 1-2)
1. Download mohamedelsharkawy's bilingual dataset from Google Drive links above
2. Convert their pickle landmarks to your format (if different)
3. Download KArSL and/or ArabicSL-Net for additional Arabic word data
4. Merge datasets and retrain your BiLSTM model

### Phase 2: Model Benchmarking (Week 2-3)
1. Train your BiLSTM + Attention on the combined dataset
2. Train a simple Random Forest baseline (mohamedelsharkawy's approach) for comparison
3. Train a GRU model (ArSLr's approach) for comparison
4. Document accuracy results in a comparison notebook
5. Keep the best-performing model for deployment

### Phase 3: Backend Completion (Week 3-4)
1. Study mohamedelsharkawy's Flask API endpoint design
2. Implement equivalent endpoints in your FastAPI backend:
   - `POST /predict/letter` — Single frame letter recognition
   - `POST /predict/word` — Video sequence word recognition
   - `WebSocket /ws/realtime` — Real-time camera stream
3. Load your `.h5` models in the FastAPI backend
4. Test with Postman or Swagger UI

### Phase 4: Frontend & Mobile (Week 4-6)
1. Complete your React web frontend with camera integration
2. Convert models to `.tflite` for mobile
3. Complete React Native mobile app
4. Test end-to-end: Camera → MediaPipe → Model → Prediction → Display

### Phase 5: Cloud Deployment (Week 6-7)
1. Dockerize the FastAPI backend
2. Deploy to Railway or similar cloud platform
3. Deploy web frontend to Vercel
4. Test mobile app against cloud API

---
## Useful Links & Resources

### Similar Projects (GitHub)
| Project | Link | Languages |
|---|---|---|
| mohamedelsharkawy (Best Match) | [GitHub](https://github.com/mohamedelsharkawy-coder/English-Arabic-Sign-Language-Recognition-Project) | English + Arabic |
| abdelrhmanmousa | [GitHub](https://github.com/abdelrhmanmousa/Arabic-Sign-Language-recognition) | Arabic |
| ArSLr (mahmoudmhashem) | [GitHub](https://github.com/mahmoudmhashem/ArSLr) | Arabic |
| AhmedDesouki | [GitHub](https://github.com/AhmedDesouki/Arabic-Sign-Language-Recognition-using-Deep-Learning-Approaches) | Arabic |

### Datasets
| Dataset | Link | Type |
|---|---|---|
| mohamedelsharkawy's data | [Google Drive](https://drive.google.com/drive/folders/1KavCk5ktwkwO3dUtO03pRv-4dJTWb6ly?usp=sharing) | English + Arabic landmarks |
| KArSL (502 words) | [Website](https://hamzah-luqman.github.io/KArSL/) | Arabic words (video) |
| ArabicSL-Net (307 words) | [Zenodo](https://zenodo.org/records/7771372) | Arabic words (video) |
| WLASL (ASL words) | [GitHub](https://github.com/dxli94/WLASL) | English words (video) |

### GitHub Topics to Follow
- [sign-language-datasets](https://github.com/topics/sign-language-datasets)
- [sign-language-recognition-system](https://github.com/topics/sign-language-recognition-system)

---

## 🏁 Key Takeaway

> **Your project architecture (BiLSTM + Attention + MobileNetV2 + FastAPI + React + React Native) is the MOST advanced and complete among all similar projects found.** The main gaps are: (1) more training data, and (2) completing the deployment pipeline. Use the datasets and API patterns from the similar projects above to fill these gaps — but keep your superior model architecture.

---

*This research was compiled on 2026-02-25. Check the linked repositories periodically for updates.*
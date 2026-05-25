# Thesis Skeleton — Bilingual Sign Language Recognition System

> **Purpose of this file.** A complete chapter-by-chapter scaffold for the FYP report,
> following the AASTMT *FYP Template Final - Individual Report.pdf* and the
> *thesis-requirements-complete-guide.pdf* in `Example Thesis/`. Every section below
> tells you (1) what to write, (2) which of your four models / full-stack components
> the section maps to, and (3) which figures, tables, or equations to insert.
>
> **Target length:** 60–80 pages of main body (Chapters 1–8), plus pre-text and
> appendices, per the AASTMT B.Sc. requirement.
>
> **Formatting recap (from the template):**
> - Body: Times New Roman 12 pt, justified, 1.5 line spacing, 12 pt before paragraph.
> - Heading 1: TNR bold 16 pt, ALL CAPS, centred.
> - Heading 2: TNR bold 14 pt, left, ALL CAPS.
> - Heading 3: TNR bold 14 pt, left, Title Case.
> - Figure captions: TNR bold 10 pt, centred, *below* the figure.
> - Table titles: TNR bold 12 pt, left, *above* the table.
> - Figure/Table numbering: `Chapter–Number` (e.g., `Figure 3-1`, `Table 5-2`).
> - References: IEEE numbered, in citation order, 50–80 entries minimum.

---

## PROPOSED PROJECT TITLE

**Eshara — A Bilingual Sign Language Recognition and Communication System
for Arabic and English Using Deep Learning**

*Subtitle:* Real-Time Letter and Word Recognition with a Cross-Platform
Web and Mobile Deployment.

**Keywords:** Sign language recognition; Arabic sign language; American sign
language; MediaPipe; BiLSTM; temporal attention; MobileNetV2; FastAPI;
React; React Native; TensorFlow Lite.

---

# A. PRE-TEXT PAGES

## A.1 Title Page
Fill in the template fields exactly: AASTMT logo, college, department, degree,
project title (all caps, bold), subtitle, your name (italic), supervisor(s),
Month–Year footer.

## A.2 Declaration
Sign and date. Confirm no plagiarism.

## A.3 Dedication (optional)
One short paragraph, centred.

## A.4 Acknowledgment
Supervisor, faculty members, family, anyone who provided datasets (KArSL
authors at KSU, Kaggle ASL Alphabet uploader, How2Sign team, MediaPipe
team), and tool providers (Kaggle for free GPU credits).

## A.5 Abstract (300–500 words)
Cover **all** of these in one continuous text:
- **Overview & background:** ~466M people worldwide and ~7M in the Arab
  region have disabling hearing loss; existing automatic sign-language
  systems focus almost exclusively on English (ASL), creating an accessibility
  gap for Arabic (ArSL) speakers.
- **Main aim & objectives:** Build a real-time bilingual SLR system covering
  both letters (finger-spelling) and words (continuous gestures), accessible
  from a web browser and a mobile phone.
- **Engineering standards used:** IEEE 802.11/HTTPS for transport;
  WebSocket (RFC 6455); REST (RFC 9110); ISO/IEC 25010 for usability;
  W3C ARIA for accessibility; TensorFlow SavedModel & TFLite formats.
- **Design considerations:** lightweight (mobile-deployable), low-latency
  (<150 ms inference), offline-capable (TFLite on device), and language-
  agnostic UI (i18n, RTL for Arabic).
- **Constraints:** limited Arabic word datasets (only KArSL-502 with 8
  samples per class), MX150 2 GB VRAM training hardware, free-tier
  cloud deployment limits.
- **Analysis & verification:** Top-1 / Top-5 accuracy, confusion matrices,
  ablation studies, real-time FPS benchmarks, user-acceptance testing.
- **Field of applications:** assistive communication, education, telemedicine,
  customer service, public-service kiosks.
- **Key numbers (fill after experiments):**
  *ASL Letter — XX % top-1; ArSL Letter — XX % top-1;
  ASL Word — XX % top-1, XX % top-5; ArSL Word — XX % top-1, XX % top-5;
  Web latency — XX ms; mobile FPS — XX.*

## A.6 Table of Contents, List of Figures, List of Tables, List of Acronyms
Auto-generated. Acronyms to define on first use **and** list here:
ASL, ArSL, SLR, CNN, RNN, LSTM, BiLSTM, GRU, MLP, MHA, BN, GAP,
MediaPipe, KArSL, WLASL, FPS, REST, WS, JWT, TFLite, ONNX, CORS,
SSE, RTL, i18n, ABET, EAB, AASTMT.

---

# B. MAIN BODY

---

## CHAPTER 1 — INTRODUCTION  *(target 8–12 pages)*

**Style:** non-technical, funnel approach (broad → narrow).

### 1.1 Background and Motivation
- Hearing impairment global and regional statistics (cite WHO 2024).
- Why sign language matters: visual–spatial grammar, not just gestures.
- The bilingual problem in the Arab world: many deaf citizens use ArSL but
  consume content/services designed for hearing speakers of Arabic or
  English. Discuss the cultural and educational angle.
- Why automated SLR is needed (24/7 accessibility, scale, cost vs human
  interpreters).

### 1.2 Problem Statement
- Existing ASL systems are mature, but ArSL is severely under-served.
- Most systems do **either** letters **or** words, not both.
- Most are research demos, not deployed products usable on a phone.

### 1.3 Aim and Objectives
**Aim.** Design, implement, evaluate, and deploy a real-time bilingual SLR
system that recognises ASL/ArSL letters and words from a single webcam or
phone camera.

**Objectives.**
1. Curate / prepare four datasets (ASL letters, ArSL letters, ASL words,
   KArSL-502 Arabic words).
2. Train four production models with consistent input pipelines.
3. Build a unified FastAPI inference backend with REST + WebSocket APIs.
4. Build a React web client with browser-side MediaPipe and a React-Native
   mobile client with on-device TFLite.
5. Evaluate accuracy, latency, and usability; compare against state-of-the-art.

### 1.4 Scope and Limitations
- **In scope:** static + dynamic gesture classification, isolated word
  recognition, real-time inference.
- **Out of scope:** continuous sentence-level sign-to-text (left as future
  work, see Chapter 8 and the `how2sign` prototype notebooks); facial-
  expression-driven grammar; signer-independent generalisation beyond
  the KArSL signers.

### 1.5 Engineering Standards & Design Considerations
List the standards from the abstract and elaborate one paragraph each:
HTTPS/WSS, REST (RFC 9110), WebSocket (RFC 6455), W3C ARIA,
ISO/IEC 25010, TensorFlow SavedModel/TFLite, IEEE software-engineering
documentation conventions.

### 1.6 Constraints
- **Economic:** free-tier cloud (Vercel + Railway), MX150 2 GB GPU.
- **Social/ethical:** signer-data consent, on-device option for privacy.
- **Environmental:** lightweight models → lower power, fewer GPU hours.
- **Technical:** TF 2.10 / Python 3.9 / Node 20 LTS lock for cuDNN-LSTM.

### 1.7 Research Contributions (Novelty)
1. **First** open-source bilingual (ASL + ArSL) letter+word system with a
   unified inference backend.
2. A reproducible MediaPipe pose-and-hands pipeline (258-feature vector)
   that runs end-to-end from raw `.mp4` to deployed model.
3. A custom `TemporalAttention` layer that improves Top-5 accuracy over
   plain BiLSTM (see Chapter 5 ablation).
4. A 3-tier deployment (FastAPI / React / Expo) demonstrating that the
   models can be served on both a cloud server and a phone.

### 1.8 Thesis Organisation
One paragraph per remaining chapter — what the reader will find.

---

## CHAPTER 2 — LITERATURE REVIEW  *(target 12–18 pages, 50+ citations)*

### 2.1 Sign Language Recognition: A Brief History
- Glove-based / sensor approaches (1990s–2000s).
- Vision-based RGB approaches.
- Skeleton-/keypoint-based deep learning (2018+).

### 2.2 Computer Vision Approaches
#### 2.2.1 Image-classification CNNs (MobileNetV2, ResNet, EfficientNet)
Cite Howard 2017, Sandler 2018 (MobileNetV2), Tan 2019 (EfficientNet).
Tie to your `MobileNetV2_Training.ipynb` and `Production_Architecture_*` notebooks.

#### 2.2.2 Keypoint/skeleton methods
Cite MediaPipe Hands (Zhang 2020), MediaPipe Holistic, OpenPose
(Cao 2017). Justify why 21 × 3 hand keypoints are sufficient for
finger-spelling and why pose + hands (258 dim) is needed for word signs.

#### 2.2.3 Hybrid (CNN + RNN)
Camgöz 2018 (Neural SLT), Koller 2020 weakly-supervised CSLR.

### 2.3 Deep-Learning Architectures
#### 2.3.1 Feed-forward / MLP (your letter models)
#### 2.3.2 Recurrent — LSTM, BiLSTM, GRU
Hochreiter 1997, Schuster 1997 (bidirectional). Equations of LSTM cell.
#### 2.3.3 Attention & Transformers
Bahdanau 2015 additive attention (the inspiration for your
`TemporalAttention` layer), Vaswani 2017 multi-head attention.

### 2.4 Pose-Estimation Frameworks
- MediaPipe Hands (33-pose + 21-hand model).
- BlazePose.
- Comparison table — speed (FPS), accuracy, ease of browser deployment.

### 2.5 Existing Sign-Language Datasets
| Dataset | Language | Type | Size | Used in this work? |
| --- | --- | --- | --- | --- |
| ASL Alphabet (Akash) | ASL | letters/images | 87 k | Yes |
| ArASL2018 | ArSL | letters/images | 54 k | Yes |
| WLASL-100/2000 | ASL | isolated words | 21 k clips | Yes (subset) |
| KArSL-502 | ArSL | isolated words | ~24 k clips | Yes |
| How2Sign | ASL | continuous | 80 h | Future work |
| RWTH-PHOENIX-Weather | DGS | continuous | 7 k sentences | Comparison only |

### 2.6 Existing Bilingual / Multilingual Systems
There are very few — cite the handful that exist (Tagliasacchi 2022,
Sidig 2021 KArSL paper) and explain how Eshara differs.

### 2.7 Real-Time SLR Deployment in the Literature
Mostly research demos; rarely deployed. Cite any web-/mobile-deployed
SLR papers you find.

### 2.8 Research Gap & Justification
Conclude the chapter by listing the gaps Eshara fills (bilingual + letters &
words + deployed + open source).

---

## CHAPTER 3 — METHODOLOGY & SYSTEM DESIGN  *(target 15–20 pages)*

### 3.1 High-Level System Architecture
**Insert Figure 3-1:** block diagram with three layers
*(use a `mermaid` flowchart converted to PNG):*

```
Camera (webcam / phone)
      │
      ▼
MediaPipe Hands / Holistic  ──►  feature vector (63 or 258)
      │
      ├─► Letter Model (MLP)      ──► letter stream decoder
      │
      └─► Word Model (BiLSTM+Att) ──► sentence builder
                                  │
                          REST / WebSocket
                                  │
                       React web   |   Expo mobile (TFLite)
```

### 3.2 Datasets
For **each** of the four datasets, write a sub-section:
- 3.2.1 ASL Alphabet — 29 classes (A–Z + space + del + nothing).
- 3.2.2 ArSL Alphabet (ArASL2018) — 32 classes.
- 3.2.3 WLASL / unified ASL Word set — 157 classes.
- 3.2.4 KArSL-502 — 502 Arabic word classes (filtered subset).
- 3.2.5 Pre-processing pipeline (frame sampling, normalisation, train/val/test
  split = 60 / 20 / 20 for words, 80 / 20 for letters).
- 3.2.6 Augmentation (Gaussian noise σ = 0.005, ±3-frame shift, 10 %
  frame-dropout, ±10 % scale, horizontal flip with L/R hand swap — pull
  the exact code from `ArSL_Word_Training_v2.ipynb` Cell 9).

**Insert Table 3-1:** dataset summary.

### 3.3 Feature Extraction
#### 3.3.1 MediaPipe Hands (letter pipeline)
21 landmarks × 3 (x,y,z) = **63** features per frame.
Wrist-centred normalisation, scale by max distance.

#### 3.3.2 MediaPipe Pose + Hands (word pipeline)
33 × 4 + 21 × 3 × 2 = **258** features per frame.
Show the index ranges: pose [0:132] | left-hand [132:195] | right-hand
[195:258].

#### 3.3.3 Temporal sampling
- Letters: single frame.
- ASL words: 30-frame window.
- ArSL words v2: 48-frame window (justify the increase — longer signs).

### 3.4 Model 1 — ASL Letter MLP
- Input: (63,)
- Architecture: Dense(256, ReLU) → Dropout → Dense(128, ReLU) → Dropout
  → Dense(29, softmax).
- Loss: categorical cross-entropy.
- File: `asl_mediapipe_mlp_model.h5`.

### 3.5 Model 2 — ArSL Letter MLP
- Same family as Model 1; trained on ArASL2018.
- File: `arsl_mediapipe_mlp_model_final.h5`.

### 3.6 Model 3 — ASL Word BiLSTM + TemporalAttention
- Input: (30, 63).
- Architecture: BiLSTM(128) → BiLSTM(64) → TemporalAttention → Dense(157).
- The **custom layer** (paste & explain mathematically):

\[
e_t = \tanh(W h_t + b), \qquad
\alpha_t = \frac{\exp(e_t)}{\sum_{t'} \exp(e_{t'})}, \qquad
c = \sum_t \alpha_t h_t .
\]

- File: `asl_word_lstm_model_best.h5` + `TemporalAttention` custom op.

### 3.7 Model 4 — ArSL Word v2 (Improved Architecture)
Take the full description from `ArSL_Word_Training_v2.ipynb`, Cell 9.
- Input: (48, 258).
- TimeDistributed Dense(192) → BN → TimeDistributed Dense(128) → BN
- Bi-LSTM(128, ret_seq) → BN → SpatialDropout(0.4)
- Bi-LSTM(96,  ret_seq) → BN → SpatialDropout(0.4)
- LSTM(64) → BN → Dropout(0.4)
- Dense(384) → BN → Dropout(0.4) → Dense(192) → Dropout(0.2)
- Softmax(num_classes).
- Training: Adam + Cosine Annealing with warm restarts, label smoothing
  0.1, gradient clipping 1.0, balanced class weights clipped to [0.5, 10].

**Insert Figure 3-2:** stacked block diagram of Model 4.

### 3.8 Engineering Trade-offs in Model Design
- Hands-only vs pose+hands.
- BiLSTM vs Transformer (memory/VRAM).
- MLP vs MobileNetV2 for letters (you tried both — give the comparison).

### 3.9 Evaluation Metrics
Formal definitions (use equations) of:
Accuracy, Top-K Accuracy, Macro-F1, Confusion matrix, Latency, FPS,
WebSocket round-trip time.

---

## CHAPTER 4 — IMPLEMENTATION  *(target 12–15 pages)*

### 4.1 Development Environment
| Layer | Stack | Version |
| --- | --- | --- |
| Training | TensorFlow / Keras + CUDA 11.2 + cuDNN 8.1 | TF 2.10 |
| Backend | FastAPI + Uvicorn + Pydantic | 0.104+ / 0.24+ |
| Web | Vite + React + TypeScript + Tailwind + MediaPipe Hands | React 18 / TS 5 |
| Mobile | Expo + React Native + TFLite + Tasks Vision | RN 0.73 |
| Auth/Chat | Node 20 + Express + Prisma + SQLite | — |
| CI/Deploy | Docker, Railway, Vercel | — |

### 4.2 Machine-Learning Pipeline
- 4.2.1 Notebook-driven workflow (`Words/ArSL Word (Arabic)/ArSL_Word_Training_v2.ipynb`, etc.).
- 4.2.2 `.npz` cache (instant restarts after first extraction).
- 4.2.3 GPU detection & mixed-precision toggle.
- 4.2.4 Model export (`.h5` → `.tflite` via `scripts/convert_models.py`).

### 4.3 Backend API (FastAPI)
Layout (from `Separated Pipelines/backend/app/`):
```
backend/app/
  main.py
  config.py
  models/
    loader.py
    letter_predictor.py
    word_predictor.py
    mode_detector.py
    letter_decoder.py     # port of letter_stream_decoder.py
    word_decoder.py
  routes/
    predict.py            # POST /predict/letter, /predict/word
    websocket_route.py    # WS  /ws/recognize
    health.py             # GET /health
  schemas/
    prediction_request.py
    prediction_response.py
    websocket_message.py
```
Discuss:
- 4.3.1 Loading 4 models at startup (memory footprint).
- 4.3.2 Letter-stream decoder rules (stability window, cooldown,
  majority vote) — paste the algorithm pseudocode.
- 4.3.3 Word predictor sliding window.
- 4.3.4 Mode detector (motion threshold switches letter ↔ word).
- 4.3.5 WebSocket protocol — JSON message schema, error handling,
  back-pressure.
- 4.3.6 CORS allow-list and security headers.

### 4.4 Web Frontend (React)
Layout (from `web/Eshara-web-main/` and `frontend/senior-main/.../frontend/`):
```
frontend/
  src/
    pages/    LandingPage.jsx  AppHomePage.jsx
    components/ Camera.tsx HandOverlay.tsx PredictionDisplay.tsx
                SentenceBuilder.tsx LanguageToggle.tsx ProtectedRoute.jsx
    hooks/    useMediaPipe.ts  useWebSocket.ts
    services/ api.js
    context/  AuthContext.jsx
```
Discuss browser MediaPipe Hands, canvas overlay drawing, WebSocket
client lifecycle, sentence-builder state machine, EN/AR i18n with
RTL flipping.

### 4.5 Authentication & Chat (Node.js layer)
Layout (from `frontend/senior-main/.../backend-api/`):
```
backend-api/
  src/
    routes/         auth.routes.js  predict.routes.js  health.routes.js
    controllers/    auth.controller.js  users.controller.js
                    predict.controller.js
    middleware/     auth.middleware.js  (JWT verify)
    sockets/        chat.socket.js
    utils/          jwt.js
  prisma/
    schema.prisma   migrations/
```
Discuss why a separate Node layer (JWT, user store, chat persistence with
Prisma + SQLite) sits between the React app and the Python ML service.

### 4.6 Mobile App (React Native + Expo + TFLite)
Layout (from `mobile/`):
```
mobile/
  App.tsx
  src/
    screens/    HomeScreen.tsx  RecognizeScreen.tsx  SettingsScreen.tsx
    components/ CameraView.tsx  ResultOverlay.tsx
    services/   tfliteModel.ts  mediapipe.ts  decoder.ts
    utils/      landmarks.ts
  assets/models/  *.tflite + label JSON
```
Discuss on-device TFLite inference (zero network round-trip), and
the JS port of the letter/word decoders for offline mode.

### 4.7 Deployment
- 4.7.1 Backend Dockerfile + Railway `railway.toml`.
- 4.7.2 Web `vercel.json`, env-var wiring (`VITE_API_URL`, `VITE_WS_URL`).
- 4.7.3 `docker-compose.yml` for local dev.
- 4.7.4 Build pipeline diagram.

**Insert Figure 4-1:** full deployment topology (web ↔ Vercel CDN ↔ users;
mobile ↔ stores ↔ users; both ↔ Railway backend ↔ models).

### 4.8 Integration Testing
- Unit tests on decoders.
- End-to-end smoke test (`webcam_test.py`).
- Lighthouse / Web Vitals on the web client.

---

## CHAPTER 5 — RESULTS AND EVALUATION  *(target 10–15 pages)*

### 5.1 Letter-Recognition Results
- 5.1.1 ASL Letter MLP — training curves, test accuracy, **confusion matrix**.
- 5.1.2 ArSL Letter MLP — same.
- 5.1.3 Per-class precision/recall tables (Table 5-1, Table 5-2).

### 5.2 Word-Recognition Results
- 5.2.1 ASL Word BiLSTM — Top-1, Top-5, confusion matrix on 157 classes.
- 5.2.2 ArSL Word v2 — Top-1, Top-5, confusion matrix on N classes.
- 5.2.3 Validation/learning-rate plots from `History` callback.

### 5.3 Comparative Analysis
Table comparing your numbers against papers listed in Chapter 2
(e.g., Sidig 2021 KArSL baseline, Camgöz BiLSTM baseline).

### 5.4 Ablation Studies *(this is critical for a high mark)*
- 5.4.1 Sequence length (30 vs 48 frames).
- 5.4.2 Feature set (hands-only 63 vs pose+hands 258).
- 5.4.3 Architecture ablation (no attention / no spatial encoder / no augment).
- 5.4.4 Augmentation impact.

### 5.5 Real-Time Performance
- End-to-end latency (camera capture → on-screen prediction).
- FPS on (a) laptop CPU, (b) laptop GPU MX150, (c) Android phone (TFLite).
- WebSocket throughput vs HTTP POST.
- Effect of mode detector on perceived smoothness.

### 5.6 Usability Evaluation
- Small user study (5–10 testers): SUS questionnaire scores.
- Qualitative feedback summary.

---

## CHAPTER 6 — DISCUSSION  *(target 8–12 pages)*

### 6.1 Interpretation of the Results
What the numbers mean, what surprised you.

### 6.2 Strengths of the Approach
- Unified pipeline across 4 models / 2 languages / 2 levels (letter, word).
- Deployment across web and mobile from one trained backbone.
- Open-source, reproducible.

### 6.3 Failure Cases & Limitations
- Visually similar signs (e.g., Arabic ك vs ل).
- Lighting / camera-distance sensitivity.
- KArSL: 8 samples/class → variance.
- MX150 VRAM limited model size.

### 6.4 Comparison with Existing Bilingual SLR
- Most existing systems: one language only, or letters only.
- You go further: bilingual + dual-level + deployed.

### 6.5 Engineering Trade-offs Revisited
- Server-side inference (cloud cost) vs on-device TFLite (model size).
- Browser MediaPipe (privacy, no upload) vs server MediaPipe (consistency).
- BiLSTM vs Transformer (chosen for VRAM, simpler).

### 6.6 Ethical, Social & Environmental Considerations
- Inclusivity gains for ArSL users.
- Privacy — landmarks instead of raw video can be transmitted.
- Energy footprint of training (Kaggle GPU hours).

---

## CHAPTER 7 — CONCLUSION  *(target 3–5 pages, must NOT be very short)*

### 7.1 Summary of Achievements
Restate the four trained models, the three deployed tiers, and the headline
numbers.

### 7.2 Contributions to the Field
Re-list the four contributions from §1.7 with the evidence found in
Chapters 5–6.

### 7.3 Practical Impact
Education, accessibility, telemedicine, and public-service kiosks.

---

## CHAPTER 8 — FUTURE WORK  *(target 1–2 pages)*

- 8.1 **Continuous SLR**: extend the existing `how2sign*.ipynb` and
  `How2Sign_Improved_v2.ipynb` prototypes into a full transformer-based
  sequence-to-sequence translator.
- 8.2 Larger Arabic vocabulary (beyond KArSL-502, e.g., dialect coverage).
- 8.3 Better on-device performance via quantisation-aware training.
- 8.4 Bidirectional system: sign-to-speech and speech-to-sign avatars.
- 8.5 Larger user study with deaf-community partners.

---

# C. POST-TEXT PAGES

## C.1 References  *(IEEE format, 50–80 entries, citation order)*

Suggested seed list (add ≥ 50):
1. M. Sandler *et al.*, "MobileNetV2: Inverted residuals and linear bottlenecks,"
   *Proc. IEEE CVPR*, 2018.
2. F. Zhang *et al.*, "MediaPipe Hands: On-device real-time hand tracking,"
   *arXiv:2006.10214*, 2020.
3. S. Hochreiter and J. Schmidhuber, "Long short-term memory,"
   *Neural Computation*, vol. 9, no. 8, 1997.
4. M. Schuster and K. Paliwal, "Bidirectional recurrent neural networks,"
   *IEEE Trans. Signal Process.*, vol. 45, no. 11, 1997.
5. A. Vaswani *et al.*, "Attention is all you need," *NeurIPS*, 2017.
6. D. Bahdanau *et al.*, "Neural machine translation by jointly learning to
   align and translate," *ICLR*, 2015.
7. A. M. Sidig *et al.*, "KArSL: Arabic sign-language database," *ACM TALLIP*, 2021.
8. N. C. Camgöz *et al.*, "Neural sign language translation," *CVPR*, 2018.
9. Z. Cao *et al.*, "OpenPose: real-time multi-person 2D pose estimation,"
   *IEEE TPAMI*, 2019.
10. World Health Organization, "Deafness and hearing loss — Fact sheet," 2024.
11. A. G. Howard *et al.*, "MobileNets: Efficient CNNs for mobile vision
    applications," *arXiv:1704.04861*, 2017.
12. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for CNNs,"
    *ICML*, 2019.
13. TensorFlow team, "TensorFlow Lite: ML for mobile and edge devices,"
    *Online*: https://www.tensorflow.org/lite , accessed 2026.
14. FastAPI documentation, *Online*: https://fastapi.tiangolo.com , 2026.
15. React documentation, *Online*: https://react.dev , 2026.
… (add datasets, ArASL2018, WLASL, How2Sign, Kaggle ASL Alphabet,
W3C ARIA, RFC 6455, RFC 9110, ISO/IEC 25010, etc.)

## C.2 Appendices

- **A. Source-code Excerpts**
  - A.1 `TemporalAttention` Keras layer.
  - A.2 `letter_stream_decoder.py` (full 262 lines).
  - A.3 ArSL word-model build (Cell 9 of `ArSL_Word_Training_v2.ipynb`).
  - A.4 FastAPI `websocket_route.py`.
  - A.5 React `useWebSocket.ts` hook.
- **B. Dataset Samples** — labelled frames from each of the four datasets.
- **C. Detailed Hyperparameter Tables** — every value from your `Cell 3` configs.
- **D. API Documentation** — auto-generated FastAPI Swagger snapshot.
- **E. User Manual** — installing the web app, the mobile app, signing in,
   using letter / word mode, EN ↔ AR toggle.
- **F. Plan of Action / Gantt chart** — fill the template's Tentative Plan
   of Action form for Oct → July.

---

# D. SECTION-BY-SECTION MAPPING TO YOUR ACTUAL CODE

> Use this as the link between thesis text and repo evidence.

| Thesis section | Repo source |
| --- | --- |
| §3.4 ASL Letter MLP | `Letters/ASL Letter (English)/Mediapipe_Training.ipynb`, `asl_mediapipe_mlp_model.h5` |
| §3.5 ArSL Letter MLP | `Letters/ArSL Letter (Arabic)/Mediapipe_Optimized_Training.ipynb`, `arsl_mediapipe_mlp_model_final.h5` |
| §3.6 ASL Word BiLSTM | `Words/ASL Word (English)/ASL_Word_Training.ipynb`, `asl_word_lstm_model_best.h5` |
| §3.7 ArSL Word v2 | `Words/ArSL Word (Arabic)/ArSL_Word_Training_v2.ipynb` |
| §3.3.2 Pose+Hands 258-dim | Cell 3 of `ArSL_Word_Training_v2.ipynb` |
| §3.8 Augmentation | Cell 9 (`augment_sequence`) of `ArSL_Word_Training_v2.ipynb` |
| §4.3 Backend | `Separated Pipelines/backend/app/` |
| §4.4 Web client | `web/Eshara-web-main/`, `frontend/senior-main/.../frontend/` |
| §4.5 Node auth/chat | `frontend/senior-main/.../backend-api/` (Express + Prisma) |
| §4.6 Mobile | `mobile/` |
| §4.7 Deployment | `Deployment/docs/*.md` (11 guides) |
| §6.3 Limitations (KArSL 8/class) | `Words/ArSL Word (Arabic)/NPZ_Check.ipynb`, `Dataset Check.ipynb` |
| Continuous SLR (Chapter 8) | `how2sign*.ipynb`, `How2Sign_Improved_v2.ipynb` |

---

# E. WRITING-ORDER SUGGESTION

Write in this order to minimise re-work:
1. Chapter 3 (Methodology) — pull facts straight from your notebook configs.
2. Chapter 4 (Implementation) — describe the code that already exists.
3. Chapter 5 (Results) — run final experiments and paste numbers/figures.
4. Chapter 2 (Literature Review) — once you know what your contributions
   are, you know what to compare against.
5. Chapter 6 (Discussion) — easy after 2 + 5.
6. Chapter 1 (Introduction) — last, because you now know exactly what the
   thesis delivers.
7. Chapter 7 (Conclusion) + Chapter 8 (Future Work) — short, write at end.
8. Abstract — write very last.
9. Pre-text pages and Appendices — assemble at the very end.

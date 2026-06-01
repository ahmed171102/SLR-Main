# Project Context for External AI (Thesis Helper)

**Project codename:** Eshara (إشارة = “sign”)  
**Domain:** Bilingual Sign Language Recognition (SLR) for **ASL (English)** and **ArSL (Arabic)**  
**Recognition levels:** **Letters** (static, per-frame) + **Words** (dynamic, isolated word classification over a sequence)  
**Core idea:** Extract pose/hand landmarks via **MediaPipe**, then classify using lightweight deep models; expose predictions through a deployable full-stack application (API + web + mobile).

This file is written to be pasted into an external AI chat so it can help write the thesis consistently, using the same naming, shapes, and components as the repository.

---

## 1) What the thesis must look like (AASTMT format)

Use the AASTMT “FYP Template Final - Individual Report” formatting. The project already summarizes the requirements in `Current Thesis/THESIS_STRUCTURE.md`.

**Formatting recap (from the template):**
- **Body**: Times New Roman 12 pt, justified, **1.5 line spacing**, ~12 pt before paragraphs
- **Heading 1**: TNR bold 16 pt, ALL CAPS, centered
- **Heading 2**: TNR bold 14 pt, ALL CAPS, left
- **Heading 3**: TNR bold 14 pt, Title Case, left
- **Figure captions**: TNR bold 10 pt, centered, **below** figure
- **Table titles**: TNR bold 12 pt, left, **above** table
- **Numbering**: `Figure 3-1`, `Table 5-2`, etc. (Chapter–Number)
- **References**: IEEE numeric, in citation order (target 50–80 references)

**Thesis scaffold (high level):**
1) Pre-text pages (Title, Declaration, Dedication optional, Acknowledgments, Abstract)  
2) TOC + List of Figures + List of Tables + Acronyms  
3) Chapters 1–8 (Intro → Future Work)  
4) References (IEEE)  
5) Appendices (code excerpts, extra results, user manual, etc.)

**Primary planning docs (already exist):**
- `Current Thesis/THESIS_STRUCTURE.md` (chapter-by-chapter scaffold)
- `Current Thesis/THESIS_PLAN_DETAILED.md` (expanded with equations + diagrams + tables + writing guidance)
- `Current Thesis/Chapter3_Methodology.md` (Word-ready Chapter 3 prose draft)

---

## 2) Repository overview (what exists vs what’s planned)

The repository (`SLR Main`) contains ML notebooks and assets; deployment code is described in `Deployment/docs/`.

**Top-level structure (from `README.md`):**
- `Letters/`  
  - `ASL Letter (English)/` notebooks + trained models  
  - `ArSL Letter (Arabic)/` notebooks + trained models  
  - `Guides/` training/optimization guides
- `Words/`  
  - `ASL Word (English)/` notebooks + trained models  
  - `ArSL Word (Arabic)/` notebooks (includes improved v2 training notebook)  
  - `Shared/` shared vocabulary / mapping files
- `Deployment/docs/` deployment plan + inventory + tech stack
- `backend/`, `web/`, `mobile/` (described as “in progress” in docs; thesis describes the intended full-stack architecture)

**Inventory of trained assets** is documented in:
- `Deployment/docs/01_INVENTORY.md`

---

## 3) The “4 major models” (inputs, outputs, files, purpose)

> External AI should keep terminology consistent: **two letter models** + **two word models** (and optionally MobileNetV2 baseline).

### Model A — ASL Letter MLP (MediaPipe landmarks → 29 classes)
- **Task:** Static letter classification per frame (A–Z + space + delete + nothing)
- **Input:** shape **(1, 63)** = 21 hand landmarks × 3 (x,y,z)
- **Output:** softmax over **29** classes
- **Model file:** `Letters/ASL Letter (English)/asl_mediapipe_mlp_model.h5`
- **Label source:** `Letters/ASL Letter (English)/asl_mediapipe_keypoints_dataset.csv`
- **Notes:** Designed for real-time usage; lightweight.

### Model B — ArSL Letter MLP (MediaPipe landmarks → Arabic letters)
- **Task:** Static Arabic letter classification per frame
- **Input:** shape **(1, 63)**
- **Output:** softmax over Arabic letter classes (dataset-specific)
- **Model file:** `Letters/ArSL Letter (Arabic)/Final Notebooks/arsl_mediapipe_mlp_model_final.h5`
- **Label source:** `Letters/ArSL Letter (Arabic)/Final Notebooks/FINAL_CLEAN_DATASET.csv`
- **Notes:** Same landmark representation as ASL letters; different classes.

### Model C — ASL Word BiLSTM + TemporalAttention (sequence → 157 words)
- **Task:** Isolated ASL word classification
- **Input:** shape **(30, 63)** (30-frame sequence of hand landmarks)
- **Output:** softmax over **157** word classes
- **Model file:** `Words/ASL Word (English)/asl_word_lstm_model_best.h5`
- **Class mapping files:**
  - `Words/ASL Word (English)/asl_word_classes.csv` (model index → word_id; 158 rows mentioned in inventory)
  - `Words/Shared/shared_word_vocabulary.csv` (word_id → English/Arabic/category; 157 rows)
- **Custom layer required:** `TemporalAttention` (see Section 5)

### Model D — ArSL Word v2 (Improved) (sequence of pose+hands → 502 words)
- **Task:** Isolated Arabic word classification on KArSL-502
- **Input:** shape **(48, 258)** per sample (48 frames, 258-D feature vector)
- **Output:** softmax over **502** word classes
- **Training notebook:** `Words/ArSL Word (Arabic)/ArSL_Word_Training_v2.ipynb`
- **Feature vector (258 per frame):**
  - Pose: 33 × 4 (x, y, z, visibility) = 132  
  - Left hand: 21 × 3 = 63  
  - Right hand: 21 × 3 = 63  
  - Total = 132 + 63 + 63 = **258**
- **Notes:** Strong regularization due to only ~8 samples per class; sequence length increased to 48 to capture the full sign arc.

### Optional baseline — MobileNetV2 (image → 29 classes)
- **Task:** Image-based ASL alphabet classification (baseline / comparison)
- **Input:** shape **(224,224,3)**  
- **Output:** 29 classes
- **Model file:** `Letters/ASL Letter (English)/sign_language_model_MobileNetV2.h5`

---

## 4) Datasets (thesis must describe them clearly)

External AI should describe the four tasks and note the constraints for Arabic words.

**Letters:**
- ASL Alphabet dataset (Kaggle) — large image set; used for ASL letter recognition
- ArASL2018 — Arabic letter dataset; used for ArSL letter recognition

**Words:**
- ASL words — from WLASL-derived/curated set (157 classes)
- KArSL-502 — Arabic isolated word dataset; **~8 samples per class**, high class count (502), strong data scarcity

**Core feature extractors:**
- MediaPipe Hands — 21 landmarks/hand → 63-D per frame
- MediaPipe Pose / Holistic — 33 pose landmarks + hands → 258-D per frame (word v2)

---

## 5) Key custom component: `TemporalAttention` layer (must be cited in thesis)

ASL word model requires a custom Keras layer to load and run.

Defined in `Deployment/docs/01_INVENTORY.md` (must be replicated in any inference service that loads the `.h5` model):

Concept (additive attention over time):
- Compute attention logits \(e_t\) from hidden states \(h_t\)
- Normalize with softmax to get \(\alpha_t\)
- Context vector \(c = \sum_t \alpha_t h_t\)

Implementation sketch (Keras):
```python
class TemporalAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight('att_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform')
        self.b = self.add_weight('att_bias', shape=(input_shape[1], 1), initializer='zeros')
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)
```

---

## 6) Streaming decoding logic (letters → stable text)

The system does not output raw per-frame letters directly; it uses stabilization and cooldown logic.

**Utility file (existing):**
- `letter_stream_decoder.py` (262 lines) — “majority vote / stabilization / cooldown” to convert per-frame predictions into usable text.

In the thesis, describe it as a **state machine** / **temporal filter**:
- Use confidence threshold(s)
- Require K consistent frames before committing a letter
- Apply cooldown to avoid repeated letters while the hand is still

---

## 7) Full-stack system architecture (what the deployed product does)

The thesis positions the project as a deployable real-time system, not just notebooks.

**Planned tiers (from `THESIS_PLAN_DETAILED.md` and `Deployment/docs/`):**
1) **Python ML backend**: FastAPI + Uvicorn + TensorFlow 2.10  
   - REST endpoints for single prediction calls  
   - WebSocket endpoint for real-time streaming inference  
2) **Node.js auth/chat backend**: Express + Prisma + SQLite + JWT + Socket.IO  
3) **Web frontend**: Vite + React + TypeScript + Tailwind  
   - Uses MediaPipe Hands in the browser  
   - Sends 63-D landmarks to backend via WebSocket  
4) **Mobile app**: React Native (Expo)  
   - Uses TFLite on-device for offline inference (future/desired)

**Why split services?**
- ML inference service scales with compute; auth/chat scales with users and DB I/O.
- Separation improves maintainability and security boundaries.

---

## 8) Key diagrams already exported as PNG (use these in thesis)

PNG files exist in `Current Thesis/figures/` and are ready to insert into Word.

**Available figure assets:**
- `fig_2-1_slr_timeline.png`
- `fig_3-1_system_architecture.png`
- `fig_3-2_preprocessing_pipeline.png`
- `fig_3-5_asl_letter_mlp.png`
- `fig_3-6_asl_word_bilstm.png`
- `fig_3-7_arsl_word_v2.png`
- `fig_4-1_training_dataflow.png`
- `fig_4-2_websocket_sequence.png`
- `fig_4-3_letter_decoder_state.png`
- `fig_4-4_react_component_tree.png`
- `fig_4-5_jwt_auth_flow.png`
- `fig_4-6_mobile_inference.png`
- `fig_4-7_deployment_topology.png`

Sources (`.mmd`) are in `Current Thesis/figures/src/`.

---

## 9) Evaluation metrics (thesis must report these)

The project’s thesis plan expects the following metrics:

**Classification metrics:**
- Top-1 accuracy (letters + words)
- Top-5 accuracy (words)
- Macro-F1 (important for class imbalance / many classes)
- Confusion matrices (letters full; words possibly top-N subset + full in appendix)

**System metrics:**
- End-to-end latency (capture + MediaPipe + network + inference + rendering)
- FPS (real-time throughput)
- Usability (optional): SUS questionnaire

---

## 10) Key constraints (must be explicitly stated)

External AI should emphasize the “engineering constraints” because they justify architectural choices.

- **Arabic word dataset scarcity:** KArSL-502 ~8 samples/class → strong regularization, careful splits, attention to overfitting
- **Hardware constraint:** training on **NVIDIA GeForce MX150 (2GB VRAM)** + optional Kaggle GPU
- **Real-time requirement:** low latency; landmarks reduce bandwidth vs raw frames
- **Bilingual + two-level recognition:** must explain mode switching and unified UI/UX (English + Arabic; RTL concerns)

---

## 11) Current thesis-writing assets (what is already drafted)

These files already exist and can be used directly:
- `Current Thesis/Chapter3_Methodology.md` — full prose Chapter 3 draft (Word-ready) with:
  - System architecture
  - Datasets
  - Feature extraction (63-D and 258-D)
  - Model descriptions (MLP, BiLSTM + TemporalAttention, ArSL Word v2)
  - Training methodology (Adam, label smoothing, cosine annealing, gradient clipping)
  - Metrics definitions

---

## 12) Quick “external AI prompt” you can copy

Use this to force consistent output from any external AI:

> You are helping me write an AASTMT FYP thesis for project “Eshara”, a bilingual ASL/ArSL sign-language recognition system. Use IEEE numeric citations placeholders like [1], [2]. Follow Times New Roman 12pt, 1.5 spacing, AASTMT Heading styles, and chapter-based figure numbering (Figure 3-1, etc.). My system has 4 major models: ASL letter MLP (input 63 → 29 classes), ArSL letter MLP (input 63 → Arabic letters), ASL word BiLSTM with custom TemporalAttention (input 30×63 → 157 words), and ArSL word v2 (input 48×258 → 502 words using pose+hands). MediaPipe Hands provides 21 landmarks/hand; pose+hands yields 258-D per frame. There is a letter_stream_decoder for stabilized text. The product architecture includes FastAPI (REST + WebSocket), React web, React Native mobile (TFLite offline), and Node auth/chat. Write Chapter X section Y with equations, figure insertion notes referencing my PNGs in Current Thesis/figures/, and do not invent file paths not listed in my context.

---

## 13) Do-not-invent rules (important for external AI)

When writing thesis text, the AI must **not invent**:
- exact accuracies (use placeholders XX.X until final experiments)
- dataset sizes beyond what is already documented
- endpoints/routes unless you confirm actual implementation
- package versions unless present in `Deployment/docs/02_TECH_STACK.md`

If something is unknown, the AI should use a TODO marker.


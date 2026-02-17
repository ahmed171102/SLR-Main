# Letters Pipeline — Full Architecture & Data Flow

> **Team Reference** — How data moves from raw images → trained model → real-time prediction

---

## End-to-End Pipeline

```
                          ┌──────────────────────────┐
                          │     RAW IMAGE DATASET      │
                          │  ASL: 87,000 images        │
                          │  ArSL: Custom collected     │
                          └────────────┬───────────────┘
                                       │
                     ┌─────────────────┼──────────────────┐
                     │                                     │
          ┌──────────▼───────────┐           ┌─────────────▼──────────┐
          │   ASL (English)      │           │   ArSL (Arabic)         │
          │   Kaggle dataset     │           │   Custom data + CSV     │
          │   29 classes (A-Z    │           │   31 classes (28 Arabic │
          │   + del/nothing/     │           │   letters + space/del/  │
          │   space)             │           │   nothing)              │
          └──────────┬───────────┘           └─────────────┬──────────┘
                     │                                     │
          ┌──────────▼───────────────────────────────────────▼──────────┐
          │              MEDIAPIPE HAND LANDMARK EXTRACTION              │
          │                                                              │
          │  For each image:                                             │
          │  1. Read image with OpenCV                                   │
          │  2. Convert BGR → RGB                                        │
          │  3. Detect hand with MediaPipe Hands (21 landmarks)          │
          │  4. Extract (x, y, z) for each landmark                      │
          │  5. Flatten: 21 × 3 = 63 features                           │
          │  6. Skip images with no hand detection                       │
          │                                                              │
          │  Output per image: 1D array of 63 floats + label             │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                    SAVE AS CSV                                │
          │  asl_mediapipe_keypoints_dataset.csv    (English)            │
          │  arabic_final_training_data.csv          (Arabic)            │
          │                                                              │
          │  Columns: 0, 1, 2, ..., 62, label                           │
          │  (63 landmark features + 1 label column)                     │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                   PREPROCESSING                               │
          │                                                              │
          │  1. LabelEncoder: letter → class index (0..N-1)              │
          │  2. One-hot encode targets (to_categorical)                  │
          │  3. Split: 64% train / 16% val / 20% test (stratified)      │
          │  4. Optional: class weights for imbalanced data              │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                  MLP MODEL (Dense Network)                    │
          │                                                              │
          │  Input(63)                                                   │
          │    → Dense(256, ReLU, L2) → BN → Drop(0.3)                  │
          │    → Dense(128, ReLU, L2) → BN → Drop(0.25)                 │
          │    → Dense(64, ReLU)      → Drop(0.2)                       │
          │    → Dense(num_classes, Softmax)                             │
          │                                                              │
          │  Train: 20 epochs max, EarlyStopping patience=5              │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │            OPTIONAL: MobileNetV2 (Image Model)                │
          │                                                              │
          │  Trained on hand-cropped images (128×128 or 96×96)           │
          │  Used in Combined notebooks for fusion with MLP              │
          │  Lower accuracy than MLP in practice                         │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                     OUTPUT FILES                              │
          │                                                              │
          │  *_mlp_model_best.h5     ← best val_accuracy checkpoint      │
          │  *_mlp_model.h5          ← final model                       │
          │  *_MobileNetV2.h5        ← optional MobileNet model          │
          │  *.csv                   ← training data / history           │
          └──────────────────────────────────────────────────────────────┘
```

---

## Real-Time Inference Pipeline

```
         ┌───────────────────────────────┐
         │          WEBCAM FEED           │
         │     (continuous frames)        │
         └──────────────┬────────────────┘
                        │
         ┌──────────────▼────────────────┐
         │   UNFLIPPED FRAME to MediaPipe │
         │   (matches training data)      │
         └──────────────┬────────────────┘
                        │
         ┌──────────────▼────────────────┐
         │  Extract 21 landmarks (63 ft)  │
         │  NO mirroring applied          │
         └──────────────┬────────────────┘
                        │
              ┌─────────┼──────────┐
              │                    │
    ┌─────────▼────────┐  ┌───────▼──────────────┐
    │   MLP Model       │  │  MobileNet (optional) │
    │   Input: (1, 63)  │  │  Input: hand crop     │
    │   → predicted     │  │  → predicted label    │
    │     label + conf  │  │    + conf             │
    └─────────┬────────┘  └───────┬──────────────┘
              │                    │
    ┌─────────▼────────────────────▼──────────────┐
    │   FUSION (higher confidence wins)            │
    │   Or MLP-only if MobileNet disabled          │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │   STABILIZATION BUFFER (majority vote)       │
    │   10 frames, 7/10 majority required          │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │   HOLD TIME CHECK (0.8 seconds)              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │   COMMIT-ONCE-THEN-WAIT                      │
    │   Lock label → wait for hand-drop or change  │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │   FLIP FRAME for selfie-view display         │
    │   Draw UI + sentence bar                     │
    └──────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why process UNFLIPPED frames?

Training data was extracted from raw camera frames **without** `cv2.flip()`. MediaPipe landmarks from flipped frames produce different x-coordinates. To match training, inference must also use unflipped frames. The flip is applied **only** for display (selfie view).

### Why NO hand mirroring?

Some code previously mirrored right-hand x-coordinates (`landmarks[:, 0] = 1 - landmarks[:, 0]`). However, the training data was collected without this transform, so applying it at inference time creates a mismatch. Removed in all active notebooks.

### Why commit-once-then-wait instead of cooldown?

The old cooldown approach (block same letter for N seconds, then allow re-commit) caused "mmmmmooooccc" — the same letter would re-commit after the cooldown expired. The new approach locks a letter permanently until the user changes their sign or removes their hand.

---

## File Structure

```
Letters/
├── ASL Letter (English)/           ← English ASL training + inference
│   ├── Combined_Architecture.ipynb ← main notebook (MLP + MobileNet fusion)
│   ├── Mediapipe_Training.ipynb    ← MLP training + webcam
│   ├── MobileNetV2_Training.ipynb  ← MobileNet training
│   ├── asl_mediapipe_keypoints_dataset.csv  ← training data
│   ├── asl_mediapipe_mlp_model.h5           ← trained MLP
│   ├── sign_language_model_MobileNetV2.h5   ← trained MobileNet
│   └── requirements.txt
│
├── ArSL Letter (Arabic)/           ← Arabic ArSL training + inference
│   ├── Mediapipe_Final_Arabic.ipynb         ← MLP training + webcam
│   ├── Final Notebooks/                      ← cleaned production notebooks
│   │   ├── Combined_Architecture_Arabic_GPU.ipynb  ← fusion (MLP + MobileNet)
│   │   ├── Mediapipe_Final_Arabic1.ipynb           ← MLP webcam
│   │   ├── Mobilenet_Arabic_Best_Final.ipynb       ← MobileNet training
│   │   ├── arsl_mediapipe_mlp_model_final.h5
│   │   └── mobilenet_arabic_final.h5
│   └── [various training/experimental notebooks]
│
├── Arabic guide/                   ← Arabic helper scripts
│   ├── arabic_class_labels.py      ← class label definitions
│   ├── arabic_data_collection.py   ← webcam data collection
│   └── arabic_display_utils.py     ← RTL text rendering
│
├── Datasets/                       ← raw image datasets
│   ├── Asl_Sign_Data/              ← English ASL images
│   └── Dataset (ArASL)/            ← Arabic sign images
│
├── Guides/                         ← reference implementations
│   ├── letter_stream_decoder.py    ← streaming decoder utility
│   ├── PAUSE_MECHANISM.md
│   ├── WORD_BUILDING_GUIDE.md
│   └── [optimization/deployment guides]
│
├── Orignal Notebooks/              ← backup copies (unfixed)
│
└── Docs/                           ← this documentation
    ├── MODEL_SUMMARY.md
    ├── ARCHITECTURE_AND_PIPELINE.md
    ├── DATASET_GUIDE.md
    ├── CLASS_LABELS.md
    ├── LETTERS_WORDS_INTEGRATION.md
    └── TEAM_QUICKSTART.md
```

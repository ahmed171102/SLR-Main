# Datasets Guide — ASL & ArSL Letters

> **For team members** — Where to get the data, what's available, and how it's structured

---

## Dataset Status Summary

| Dataset | Language | Classes | Samples | Status | Location |
|---|---|---|---|---|---|
| **ASL Alphabet** | English | 29 | ~87,000 images | ✅ **Ready** | `Datasets/Asl_Sign_Data/` |
| **ArSL Custom** | Arabic | 31–34 | Custom collected | ✅ **Ready** | `Datasets/Dataset (ArASL)/` + CSV files |

---

## 1. ASL Alphabet Dataset (English)

**What Is It?**
- American Sign Language alphabet images
- 29 classes: A–Z + `del`, `nothing`, `space`
- ~3,000 images per class (200×200 RGB)
- Controlled background, single hand

**Download:**
- Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

**Current State:**
```
Letters/Datasets/Asl_Sign_Data/
├── asl_alphabet_train/         ← ~87,000 training images
│   ├── A/   (3,000 images)
│   ├── B/   (3,000 images)
│   ├── ...
│   ├── Z/   (3,000 images)
│   ├── del/
│   ├── nothing/
│   └── space/
└── asl_alphabet_test/          ← 29 test images (1 per class)
```

**How It's Used:**
1. Images are processed through MediaPipe to extract 21 hand landmarks
2. Landmarks are flattened to 63 features and saved in CSV
3. The MLP model trains on the CSV (not raw images)
4. MobileNetV2 trains directly on the images (128×128 crop)

**Pre-extracted CSV:**
```
Letters/ASL Letter (English)/asl_mediapipe_keypoints_dataset.csv
Columns: 0, 1, 2, ..., 62, label
         63 landmark features   class letter
```

---

## 2. ArSL Custom Dataset (Arabic)

**What Is It?**
- Arabic Sign Language letters collected via webcam
- 31 core classes: 28 Arabic letters + `space`, `del`, `nothing`
- Some models trained on expanded set (34 labels including: Al, Laa, Teh_Marbuta, Thal)

**Current State:**
```
Letters/ArSL Letter (Arabic)/
├── arabic_final_training_data.csv      ← main training CSV
├── arabic_fixed_final.csv              ← cleaned version
├── arabic_fixed_with_nothing.csv       ← with 'nothing' class added
├── arabic_ready_for_training.csv       ← ready-to-train version
└── Arabic Sign Language Letters Dataset.csv  ← original

Letters/Datasets/Dataset (ArASL)/       ← raw images (compressed)
└── Dataset (ArASL).rar
```

**Arabic Classes (31 core):**
```
ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي
+ space, del, nothing
```

**Data Collection:**
Use `Arabic guide/arabic_data_collection.py` to collect new samples via webcam. It captures MediaPipe landmarks and saves them to CSV.

---

## 3. CSV Format

Both English and Arabic training data follow this format:

| Column | Content | Type |
|---|---|---|
| 0–62 | Landmark features (21 × 3 = 63 values) | float |
| label | Class label (letter or control) | string |

**English label column:** last column, named `label`  
**Arabic label column:** first or last column depending on CSV version — check before training

**Coordinate mapping:**
```
Column 0  = Landmark 0 (WRIST) x
Column 1  = Landmark 0 (WRIST) y
Column 2  = Landmark 0 (WRIST) z
Column 3  = Landmark 1 (THUMB_CMC) x
...
Column 60 = Landmark 20 (PINKY_TIP) x
Column 61 = Landmark 20 (PINKY_TIP) y
Column 62 = Landmark 20 (PINKY_TIP) z
```

---

## 4. Important Notes

### Frame Orientation
- **Training data was collected from UNFLIPPED camera frames**
- Inference must also use unflipped frames before MediaPipe processing
- Flip only for display (selfie view)

### No Hand Mirroring
- Training data was captured without mirroring right-hand landmarks
- Do NOT apply `landmarks[:, 0] = 1 - landmarks[:, 0]` during inference

### Class Count Variations
| Source | English Classes | Arabic Classes |
|---|---|---|
| MLP model | 29 | 31 or 34 |
| MobileNet model | 29 | 35 |
| CSV unique labels | 29 | 34 |
| `arabic_class_labels.py` | — | 31 |

The mismatch in Arabic class counts is why MobileNet fusion is sometimes auto-disabled.

---

## 5. Tips for Team Members

- English dataset is large (~87K images) — MediaPipe extraction takes 30–60 min, but the CSV is pre-extracted
- Arabic data is smaller — custom collected, so quality may vary
- Don't delete CSV files — they are the pre-processed training data
- If adding new classes, update both the CSV and `arabic_class_labels.py`
- Use `arabic_data_collection.py` to collect new Arabic samples consistently

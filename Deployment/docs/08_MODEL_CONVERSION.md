# 08 — Model Conversion: .h5 → TFLite

> Convert your trained Keras models to TFLite format for mobile deployment.

---

## Why TFLite?

| Format | Size | Speed | Where |
|--------|------|-------|-------|
| `.h5` (Keras) | 10-50 MB | Normal | Backend (Python) |
| `.tflite` | 2-15 MB | 2-5x faster | Mobile phone (Android/iOS) |

TFLite models are:
- Smaller (60-80% size reduction)
- Faster inference
- Run on mobile without Python or TensorFlow full framework
- Can use phone's GPU or Neural Engine for acceleration

---

## Prerequisites

- Python 3.9 with TensorFlow 2.10.0 installed
- All .h5 model files available
- Label CSV files available

---

## Script: `scripts/convert_models.py`

Create this file and run it to convert all models:

```python
"""
Convert .h5 Keras models to TFLite format for mobile deployment.
Also exports label files as JSON.

Usage:
    cd m:\Term 10\Grad\Deployment
    python scripts/convert_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────
SLR_ROOT = Path(r"m:\Term 10\Grad\Main\Sign-Language-Recognition-System-main\SLR Main")

# Source model files
MODELS = {
    "asl_letter": SLR_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_mlp_model.h5",
    "arsl_letter": SLR_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "arsl_mediapipe_mlp_model_final.h5",
    "asl_word": SLR_ROOT / "Words" / "ASL Word (English)" / "asl_word_lstm_model_best.h5",
}

# Source label files
LABELS = {
    "asl_letter_csv": SLR_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_keypoints_dataset.csv",
    "arsl_letter_csv": SLR_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "FINAL_CLEAN_DATASET.csv",
    "word_classes_csv": SLR_ROOT / "Words" / "ASL Word (English)" / "asl_word_classes.csv",
    "word_vocab_csv": SLR_ROOT / "Words" / "Shared" / "shared_word_vocabulary.csv",
}

# Output directories
OUTPUT_TFLITE = Path(r"m:\Term 10\Grad\Deployment\mobile\assets\models")
OUTPUT_BACKEND = Path(r"m:\Term 10\Grad\Deployment\backend\model_files")


# ─── TemporalAttention (needed to load word model) ──────────────
class TemporalAttention(tf.keras.layers.Layer):
    """Custom attention layer used in word BiLSTM model."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


# ─── Helper Functions ────────────────────────────────────────────

def convert_simple_model(model_path: Path, output_path: Path, model_name: str):
    """Convert a simple MLP model (letter models)."""
    print(f"\n{'='*60}")
    print(f"Converting: {model_name}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")

    # Load model
    model = tf.keras.models.load_model(str(model_path))
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization: float16 quantization (smaller + fast on mobile GPU)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    # Size comparison
    original_size = model_path.stat().st_size / 1024 / 1024
    tflite_size = len(tflite_model) / 1024 / 1024
    reduction = (1 - tflite_size / original_size) * 100

    print(f"  Original: {original_size:.2f} MB")
    print(f"  TFLite:   {tflite_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  ✓ Saved: {output_path}")


def convert_word_model(model_path: Path, output_path: Path):
    """Convert word BiLSTM model (has custom TemporalAttention layer)."""
    print(f"\n{'='*60}")
    print("Converting: ASL Word BiLSTM")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")

    # Load with custom objects
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={'TemporalAttention': TemporalAttention}
    )
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Convert to TFLite
    # IMPORTANT: Word model uses custom ops, need SELECT_TF_OPS
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # Allow TF ops for custom attention layer
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    original_size = model_path.stat().st_size / 1024 / 1024
    tflite_size = len(tflite_model) / 1024 / 1024
    reduction = (1 - tflite_size / original_size) * 100

    print(f"  Original: {original_size:.2f} MB")
    print(f"  TFLite:   {tflite_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  ✓ Saved: {output_path}")


def export_labels():
    """Export label encoders and word mappings to JSON for mobile/web."""
    print(f"\n{'='*60}")
    print("Exporting label files to JSON...")

    # 1. ASL Letter labels
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(str(LABELS["asl_letter_csv"]))
    le = LabelEncoder()
    le.fit(df['label'])
    asl_labels = le.classes_.tolist()

    asl_path = OUTPUT_TFLITE / "asl_letter_labels.json"
    asl_path.write_text(json.dumps(asl_labels, indent=2))
    print(f"  ✓ ASL Letter Labels: {len(asl_labels)} classes → {asl_path}")

    # Also save to backend
    (OUTPUT_BACKEND / "asl_letter_labels.json").write_text(json.dumps(asl_labels, indent=2))

    # 2. ArSL Letter labels
    df = pd.read_csv(str(LABELS["arsl_letter_csv"]))
    le = LabelEncoder()
    le.fit(df['label'])
    arsl_labels = le.classes_.tolist()

    arsl_path = OUTPUT_TFLITE / "arsl_letter_labels.json"
    arsl_path.write_text(json.dumps(arsl_labels, indent=2, ensure_ascii=False))
    print(f"  ✓ ArSL Letter Labels: {len(arsl_labels)} classes → {arsl_path}")

    (OUTPUT_BACKEND / "arsl_letter_labels.json").write_text(json.dumps(arsl_labels, indent=2, ensure_ascii=False))

    # 3. Word labels (bilingual)
    word_classes = pd.read_csv(str(LABELS["word_classes_csv"]))
    word_vocab = pd.read_csv(str(LABELS["word_vocab_csv"]))

    word_labels = []
    for _, row in word_classes.iterrows():
        word_id = int(row['word_id'])
        vocab_row = word_vocab[word_vocab['word_id'] == word_id]
        if not vocab_row.empty:
            word_labels.append({
                "id": word_id,
                "model_index": int(row['model_class_index']),
                "en": str(vocab_row.iloc[0]['english']),
                "ar": str(vocab_row.iloc[0]['arabic']),
                "category": str(vocab_row.iloc[0].get('category', 'unknown'))
            })

    # Sort by model_index for direct array indexing
    word_labels.sort(key=lambda x: x['model_index'])

    word_path = OUTPUT_TFLITE / "word_labels.json"
    word_path.write_text(json.dumps(word_labels, indent=2, ensure_ascii=False))
    print(f"  ✓ Word Labels: {len(word_labels)} words → {word_path}")

    (OUTPUT_BACKEND / "word_labels.json").write_text(json.dumps(word_labels, indent=2, ensure_ascii=False))


def verify_tflite(tflite_path: Path, input_shape: tuple, model_name: str):
    """Verify a TFLite model works correctly."""
    print(f"\n  Verifying {model_name}...")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    Input:  {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"    Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")

    # Test with random data
    test_input = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"    Test prediction shape: {output.shape}")
    print(f"    Sum of probabilities: {output.sum():.4f} (should be ~1.0)")
    print(f"    ✓ Verified!")


# ─── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ESHARA Model Conversion Tool")
    print("=" * 60)

    # Create output directories
    OUTPUT_TFLITE.mkdir(parents=True, exist_ok=True)
    OUTPUT_BACKEND.mkdir(parents=True, exist_ok=True)

    # Check source files exist
    print("\nChecking source files...")
    all_exist = True
    for name, path in {**MODELS, **LABELS}.items():
        exists = path.exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}: {name} → {path.name}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠ Some source files are missing. Fix before continuing.")
        exit(1)

    # Convert models
    convert_simple_model(
        MODELS["asl_letter"],
        OUTPUT_TFLITE / "asl_letter_model.tflite",
        "ASL Letter MLP"
    )

    convert_simple_model(
        MODELS["arsl_letter"],
        OUTPUT_TFLITE / "arsl_letter_model.tflite",
        "ArSL Letter MLP"
    )

    convert_word_model(
        MODELS["asl_word"],
        OUTPUT_TFLITE / "asl_word_model.tflite"
    )

    # Export labels
    export_labels()

    # Verify
    print(f"\n{'='*60}")
    print("Verifying TFLite models...")

    verify_tflite(
        OUTPUT_TFLITE / "asl_letter_model.tflite",
        (1, 63),
        "ASL Letter"
    )

    verify_tflite(
        OUTPUT_TFLITE / "arsl_letter_model.tflite",
        (1, 63),
        "ArSL Letter"
    )

    verify_tflite(
        OUTPUT_TFLITE / "asl_word_model.tflite",
        (1, 30, 63),
        "ASL Word"
    )

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"  TFLite models: {OUTPUT_TFLITE}")
    print(f"  Backend labels: {OUTPUT_BACKEND}")
```

---

## How to Run

```powershell
cd "m:\Term 10\Grad\Deployment"

# Activate Python env (or use the one from backend)
python scripts/convert_models.py
```

Expected output:
```
ESHARA Model Conversion Tool
============================================================

Checking source files...
  ✓: asl_letter → asl_mediapipe_mlp_model.h5
  ✓: arsl_letter → arsl_mediapipe_mlp_model_final.h5
  ✓: asl_word → asl_word_lstm_model_best.h5
  ✓: asl_letter_csv → asl_mediapipe_keypoints_dataset.csv
  ...

============================================================
Converting: ASL Letter MLP
  Input shape:  (None, 63)
  Output shape: (None, 29)
  Original: 0.45 MB
  TFLite:   0.12 MB
  Reduction: 73.3%
  ✓ Saved

...

ALL DONE!
```

---

## Script: `scripts/copy_models.py`

Copy original .h5 and CSV files to the backend:

```python
"""
Copy model and label files from SLR Main into the backend model_files directory.

Usage:
    cd m:\Term 10\Grad\Deployment
    python scripts/copy_models.py
"""

import shutil
from pathlib import Path

SLR_ROOT = Path(r"m:\Term 10\Grad\Main\Sign-Language-Recognition-System-main\SLR Main")
TARGET = Path(r"m:\Term 10\Grad\Deployment\backend\model_files")

FILES_TO_COPY = [
    # Model files
    (SLR_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_mlp_model.h5", "asl_mediapipe_mlp_model.h5"),
    (SLR_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "arsl_mediapipe_mlp_model_final.h5", "arsl_mediapipe_mlp_model_final.h5"),
    (SLR_ROOT / "Words" / "ASL Word (English)" / "asl_word_lstm_model_best.h5", "asl_word_lstm_model_best.h5"),
    (SLR_ROOT / "Letters" / "ASL Letter (English)" / "sign_language_model_MobileNetV2.h5", "sign_language_model_MobileNetV2.h5"),

    # Label files
    (SLR_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_keypoints_dataset.csv", "asl_mediapipe_keypoints_dataset.csv"),
    (SLR_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "FINAL_CLEAN_DATASET.csv", "FINAL_CLEAN_DATASET.csv"),
    (SLR_ROOT / "Words" / "ASL Word (English)" / "asl_word_classes.csv", "asl_word_classes.csv"),
    (SLR_ROOT / "Words" / "Shared" / "shared_word_vocabulary.csv", "shared_word_vocabulary.csv"),
]

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)

    print("Copying model and label files...")
    for src, dest_name in FILES_TO_COPY:
        dest = TARGET / dest_name
        if src.exists():
            shutil.copy2(str(src), str(dest))
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  ✓ {dest_name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ MISSING: {src}")

    print(f"\nDone! Files copied to: {TARGET}")
```

---

## Troubleshooting

### "Custom op not found" when converting word model

The word model uses `TemporalAttention`, a custom Keras layer. Need:
```python
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS   # <-- This enables custom ops
]
```

On mobile, you may need the TFLite Flex delegate to run SELECT_TF_OPS models.

### Model too large for mobile

If the word model TFLite is still >20MB:
```python
# More aggressive quantization (may lose some accuracy)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Or full int8 quantization (smallest, but needs representative dataset)
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 30, 63).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```

### Conversion fails entirely

Fallback: Use the backend API from mobile instead of on-device inference.
Change mobile app to send landmarks via HTTP instead of using TFLite locally.

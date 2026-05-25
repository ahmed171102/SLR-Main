
# ============================================================
# ArSL MobileNetV2 - Kaggle-Ready Notebook
# Arabic Sign Language Recognition
# ============================================================

# Cell 1: Setup & Imports
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix, classification_report
from scipy import ndimage
from collections import Counter

# ─── Config ───────────────────────────────────────────────
IMG_SIZE     = 96
BATCH_SIZE   = 32
INITIAL_EPOCHS = 8
FINETUNE_EPOCHS = 10
NUM_CLASSES  = 29        # ArASL 2024: 29 letter classes
SEED         = 42

# ─── Paths (auto-detect Kaggle vs local) ──────────────────
if os.path.exists("/kaggle/input"):
    # List available datasets
    print("Kaggle datasets found:")
    for d in os.listdir("/kaggle/input"):
        print(f"  /kaggle/input/{d}")
    # Update this to your dataset slug:
    DATASET_SLUG = "arsl2021"          # <-- change if needed
    TRAIN_DIR    = f"/kaggle/input/{DATASET_SLUG}/train"
    VAL_DIR      = f"/kaggle/input/{DATASET_SLUG}/val"
    TEST_DIR     = f"/kaggle/input/{DATASET_SLUG}/test"
    OUTPUT_DIR   = "/kaggle/working"
else:
    # Local Windows paths
    BASE = r"M:\Term 9\Grad\Main\Sign-Language-Recognition-System-main\Sign-Language-Recognition-System-main\Sign_to_Sentence Project Main\Datasets\Dataset (ArASL)\ArASL Database"
    TRAIN_DIR  = os.path.join(BASE, "train")
    VAL_DIR    = os.path.join(BASE, "val")
    TEST_DIR   = os.path.join(BASE, "ArASL_35")
    OUTPUT_DIR = "."

# ─── GPU setup ────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[g.name for g in gpus]}")
else:
    print("No GPU found – running on CPU")

tf.keras.mixed_precision.set_global_policy("float32")
print(f"TF version: {tf.__version__}")

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# Cell 2: Dataset Preparation
# ============================================================

def get_class_names(directory):
    return sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f == ".DS_Store":
                os.remove(os.path.join(root, f))

# Remove .DS_Store artifacts
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(d):
        remove_ds_store(d)

class_names = get_class_names(TRAIN_DIR)
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}")

# Count samples per class
class_counts = {}
for cls in class_names:
    cls_path = os.path.join(TRAIN_DIR, cls)
    class_counts[cls] = len([
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

print("\nClass distribution (train):")
for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1]):
    print(f"  {cls:20s}: {cnt:,}")
print(f"\nTotal train images: {sum(class_counts.values()):,}")

# ============================================================
# Cell 3: tf.data Pipeline
# ============================================================

def build_dataset(directory, image_size, batch_size, augment=False):
    """Build a tf.data.Dataset from a directory."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
        shuffle=augment,
        seed=SEED,
    )

    def preprocess(images, labels):
        images = preprocess_input(images)
        return images, labels

    def augment_fn(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, max_delta=0.2)
        images = tf.image.random_contrast(images, 0.8, 1.2)
        return images, labels

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache().prefetch(AUTOTUNE)
    return ds

print("Building datasets...")
train_ds = build_dataset(TRAIN_DIR, IMG_SIZE, BATCH_SIZE, augment=True)
val_ds   = build_dataset(VAL_DIR,   IMG_SIZE, BATCH_SIZE, augment=False)
print("Datasets ready.")

# ============================================================
# Cell 4: Class Balancing via Sample Weights
# ============================================================

total = sum(class_counts.values())
class_weight = {
    i: total / (NUM_CLASSES * class_counts[cls])
    for i, cls in enumerate(class_names)
}
print("Class weights computed (top 5 by weight):")
top5 = sorted(class_weight.items(), key=lambda x: x[1], reverse=True)[:5]
for idx, w in top5:
    print(f"  {class_names[idx]:20s}: {w:.4f}")

# ============================================================
# Cell 5: Model Architecture
# ============================================================

def build_model(num_classes, img_size):
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base

model, base_model = build_model(NUM_CLASSES, IMG_SIZE)
model.summary()

# ============================================================
# Cell 6: Phase 1 – Initial Training
# ============================================================

cb_initial = [
    callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "mobilenet_arabic_best_initial.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1,
    ),
    callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_initial.csv")),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1,
    ),
]

print("=" * 70)
print("PHASE 1: INITIAL TRAINING")
print("=" * 70)
history_initial = model.fit(
    train_ds,
    epochs=INITIAL_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=cb_initial,
)

# ============================================================
# Cell 7: Phase 2 – Fine-Tuning
# ============================================================

# Unfreeze last 40 layers
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep BN layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb_finetune = [
    callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "mobilenet_arabic_best_finetuned.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1,
    ),
    callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_finetune.csv")),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1,
    ),
]

print("=" * 70)
print("PHASE 2: FINE-TUNING")
print("=" * 70)
history_finetune = model.fit(
    train_ds,
    epochs=FINETUNE_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=cb_finetune,
)

# Save final model
model.save(os.path.join(OUTPUT_DIR, "mobilenet_arabic_final.h5"))
print("Final model saved.")

# ============================================================
# Cell 8: Training Visualization
# ============================================================

def plot_training_history(h1, h2):
    hist1 = h1.history
    hist2 = h2.history
    acc     = hist1["accuracy"]     + hist2["accuracy"]
    val_acc = hist1["val_accuracy"] + hist2["val_accuracy"]
    loss    = hist1["loss"]         + hist2["loss"]
    val_loss= hist1["val_loss"]     + hist2["val_loss"]
    epochs_range = range(1, len(acc) + 1)
    split = len(hist1["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, acc,     label="Train Acc", marker="o", color="orange")
    ax1.plot(epochs_range, val_acc, label="Val Acc",   marker="o", color="red")
    ax1.axvline(x=split + 0.5, color="green", linestyle="--", label="Fine-tune Start")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss,     label="Train Loss", marker="o", color="blue")
    ax2.plot(epochs_range, val_loss, label="Val Loss",   marker="o", color="purple")
    ax2.axvline(x=split + 0.5, color="green", linestyle="--", label="Fine-tune Start")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
    plt.show()
    print(f"Best Val Accuracy: {max(val_acc):.4f}")

plot_training_history(history_initial, history_finetune)

# ============================================================
# Cell 9: TTA Evaluation on Test Set
# ============================================================

def create_tta_images(image):
    """5-crop TTA."""
    return np.array([
        image,
        ndimage.rotate(image, 5,  reshape=False, mode="nearest"),
        ndimage.rotate(image, -5, reshape=False, mode="nearest"),
        np.clip(image * 1.1, 0, 255),
        np.clip(image * 0.9, 0, 255),
    ])

# Load best model
best_model_path = os.path.join(OUTPUT_DIR, "mobilenet_arabic_best_finetuned.h5")
if os.path.exists(best_model_path):
    best_model = tf.keras.models.load_model(best_model_path)
    print("Loaded best fine-tuned model")
else:
    best_model = model
    print("Using current model")

# Load test images (one per class sub-folder)
test_images, image_names = [], []
if os.path.exists(TEST_DIR):
    for cls_name in sorted(os.listdir(TEST_DIR)):
        cls_path = os.path.join(TEST_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            img = cv2.imread(os.path.join(cls_path, imgs[0]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
                test_images.append(img)
                image_names.append(cls_name)

test_images = np.array(test_images)
print(f"Loaded {len(test_images)} test images")

# Run TTA predictions
all_preds = []
for img in test_images:
    tta_imgs = create_tta_images(img)
    tta_imgs_pre = preprocess_input(tta_imgs.copy())
    preds = best_model.predict(tta_imgs_pre, verbose=0, batch_size=5)
    all_preds.append(np.mean(preds, axis=0))

all_preds = np.array(all_preds)
pred_classes = np.argmax(all_preds, axis=1)
pred_confs   = np.max(all_preds, axis=1)

correct = sum(image_names[i] == class_names[pred_classes[i]]
              for i in range(len(image_names)))
accuracy = correct / len(image_names) * 100 if image_names else 0

print(f"\nTest Accuracy (TTA): {accuracy:.2f}%")
print(f"Average Confidence: {pred_confs.mean()*100:.1f}%")

# ============================================================
# Cell 10: Confusion Matrix
# ============================================================

pred_labels = [class_names[pred_classes[i]] for i in range(len(image_names))]
label_set   = sorted(set(class_names))
cm = confusion_matrix(list(image_names), pred_labels, labels=label_set)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_set, yticklabels=label_set)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix – Test Set (with TTA)", fontweight="bold")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# ============================================================
# Cell 11: Final Summary
# ============================================================

print("=" * 70)
print("FINAL SUMMARY – ArSL MobileNetV2 (Kaggle Edition)")
print("=" * 70)
print(f"  Classes:          {NUM_CLASSES}")
print(f"  Image size:       {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size:       {BATCH_SIZE}")
print(f"  Initial epochs:   {INITIAL_EPOCHS}")
print(f"  Finetune epochs:  {FINETUNE_EPOCHS}")
print(f"  Test Accuracy:    {accuracy:.2f}%")
print("=" * 70)
print("Saved files:")
for fname in ["mobilenet_arabic_best_initial.h5",
              "mobilenet_arabic_best_finetuned.h5",
              "mobilenet_arabic_final.h5",
              "training_initial.csv",
              "training_finetune.csv",
              "training_history.png",
              "confusion_matrix.png"]:
    full = os.path.join(OUTPUT_DIR, fname)
    exists = "✓" if os.path.exists(full) else "✗"
    print(f"  {exists} {fname}")

import json


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# ArSL MobileNetV2 — Kaggle Edition\n\n"
    "Arabic Sign Language recognition | 32 classes | 190,000 images | MobileNetV2\n\n"
    "**Pipeline:** tf.data + AUTOTUNE | Two-phase training | TTA inference"
))

# ── 1. Setup & Imports ────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Setup & Imports"))
cells.append(code_cell(
    "import os\n"
    "import cv2\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "import matplotlib.pyplot as plt\n"
    "import tensorflow as tf\n"
    "from tensorflow.keras.applications import MobileNetV2\n"
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n"
    "from tensorflow.keras import layers, models, callbacks\n"
    "from sklearn.metrics import confusion_matrix\n"
    "from scipy import ndimage\n"
    "\n"
    "# ── Hyperparameters ─────────────────────────────────────────────────────\n"
    "IMG_SIZE        = 96\n"
    "BATCH_SIZE      = 32\n"
    "INITIAL_EPOCHS  = 8\n"
    "FINETUNE_EPOCHS = 10\n"
    "SEED            = 42\n"
    "VAL_SPLIT       = 0.2   # used only when dataset has no train/val folders\n"
    "\n"
    "OUTPUT_DIR   = '/kaggle/working' if os.path.exists('/kaggle/input') else '.'\n"
    "DATASET_MODE = None   # set to 'split' or 'flat' below\n"
    "DATA_ROOT    = None\n"
    "TRAIN_DIR    = None\n"
    "VAL_DIR      = None\n"
    "TEST_DIR     = None\n"
    "\n"
    "# ── GPU setup ───────────────────────────────────────────────────────────\n"
    "gpus = tf.config.list_physical_devices('GPU')\n"
    "for gpu in gpus:\n"
    "    tf.config.experimental.set_memory_growth(gpu, True)\n"
    "tf.keras.mixed_precision.set_global_policy('float32')\n"
    "gpu_names = [g.name for g in gpus] if gpus else ['None - CPU']\n"
    "print(f'TF {tf.__version__} | GPUs: {gpu_names}')\n"
    "AUTOTUNE = tf.data.AUTOTUNE\n"
))

# ── 2. Labels & Counts ────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 2. Dataset Labels & Class Counts\n\n"
    "Hardcoded from `Number_of_images_per_Letter.csv` (32 classes, 190,000 images)."
))
cells.append(code_cell(
    "# 32 Arabic letter classes — exact counts from Number_of_images_per_Letter.csv\n"
    "KNOWN_COUNTS = {\n"
    "    'ain': 5448, 'al': 5250, 'aleff': 5897, 'bb': 5380,\n"
    "    'dal': 5227, 'dha': 5995, 'dhad': 6326, 'fa': 6858,\n"
    "    'gaaf': 5630, 'ghain': 5850, 'ha': 6698, 'haa': 7092,\n"
    "    'jeem': 6456, 'kaaf': 6435, 'khaa': 6679, 'la': 5784,\n"
    "    'laam': 5510, 'meem': 5275, 'nun': 5760, 'ra': 5612,\n"
    "    'saad': 5723, 'seen': 5139, 'sheen': 5240, 'ta': 6166,\n"
    "    'taa': 6081, 'thaa': 6083, 'thal': 6805, 'toot': 6651,\n"
    "    'waw': 5977, 'ya': 6354, 'yaa': 5511, 'zay': 5108,\n"
    "}\n"
    "\n"
    "class_names = sorted(KNOWN_COUNTS.keys())\n"
    "NUM_CLASSES  = len(class_names)\n"
    "\n"
    "print(f'Classes ({NUM_CLASSES}): {class_names}')\n"
    "print(f'Total images: {sum(KNOWN_COUNTS.values()):,}')\n"
    "for cls in class_names:\n"
    "    print(f'  {cls:10s}: {KNOWN_COUNTS[cls]:,}')\n"
))

# ── 3. Path Detection ─────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 3. Path Detection\n\n"
    "Auto-detects Kaggle dataset structure — works for both:\n"
    "- **Split mode**: dataset has `train/` and `val/` subfolders\n"
    "- **Flat mode**: dataset has class folders directly (80/20 split applied automatically)"
))
cells.append(code_cell(
    "if os.path.exists('/kaggle/input'):\n"
    "    print('=== /kaggle/input structure (up to depth 4) ===')\n"
    "    for _r, _d, _f in os.walk('/kaggle/input'):\n"
    "        _level = _r.replace('/kaggle/input', '').count(os.sep)\n"
    "        if _level > 4:\n"
    "            continue\n"
    "        _indent = '  ' * _level\n"
    "        print(f'{_indent}{os.path.basename(_r)}/')\n"
    "    print('=' * 50)\n"
    "\n"
    "    # Try 1: find folder with pre-made train/ and val/ subdirectories\n"
    "    for _root, _dirs, _ in os.walk('/kaggle/input'):\n"
    "        if 'train' in _dirs and 'val' in _dirs:\n"
    "            DATA_ROOT    = _root\n"
    "            TRAIN_DIR    = os.path.join(_root, 'train')\n"
    "            VAL_DIR      = os.path.join(_root, 'val')\n"
    "            TEST_DIR     = os.path.join(_root, 'test')\n"
    "            DATASET_MODE = 'split'\n"
    "            print('Mode: SPLIT  (train/val folders exist)')\n"
    "            print('TRAIN_DIR: ' + TRAIN_DIR)\n"
    "            print('VAL_DIR  : ' + VAL_DIR)\n"
    "            break\n"
    "\n"
    "    # Try 2: find folder whose direct subfolders are class names (flat structure)\n"
    "    if DATASET_MODE is None:\n"
    "        for _root, _dirs, _ in os.walk('/kaggle/input'):\n"
    "            if len(set(_dirs) & set(class_names)) >= 20:\n"
    "                DATA_ROOT    = _root\n"
    "                DATASET_MODE = 'flat'\n"
    "                print('Mode: FLAT  (class folders directly in root)')\n"
    "                print('DATA_ROOT: ' + DATA_ROOT)\n"
    "                break\n"
    "\n"
    "    if DATASET_MODE is None:\n"
    "        raise FileNotFoundError(\n"
    "            'Cannot locate ASLAD-190K dataset under /kaggle/input.\\n'\n"
    "            'Attach the dataset to this notebook and re-run.'\n"
    "        )\n"
    "\n"
    "else:\n"
    "    # Local Windows paths\n"
    "    _BASE = (\n"
    "        r'M:\\Term 9\\Grad\\Main'\n"
    "        r'\\Sign-Language-Recognition-System-main'\n"
    "        r'\\Sign-Language-Recognition-System-main'\n"
    "        r'\\Sign_to_Sentence Project Main'\n"
    "        r'\\Datasets\\Dataset (ArASL)\\ArASL Database'\n"
    "    )\n"
    "    TRAIN_DIR    = os.path.join(_BASE, 'train')\n"
    "    VAL_DIR      = os.path.join(_BASE, 'val')\n"
    "    TEST_DIR     = os.path.join(_BASE, 'ArASL_35')\n"
    "    DATA_ROOT    = _BASE\n"
    "    DATASET_MODE = 'split'\n"
    "    OUTPUT_DIR   = '.'\n"
    "    print('Mode: SPLIT  (local)')\n"
    "    print('TRAIN_DIR: ' + TRAIN_DIR)\n"
))

# ── 4. tf.data Pipeline ───────────────────────────────────────────────────────
cells.append(md_cell("## 4. tf.data Pipeline"))
cells.append(code_cell(
    "def make_ds(directory, augment=False, subset=None, val_split=0.2):\n"
    "    kwargs = dict(\n"
    "        directory=directory,\n"
    "        image_size=(IMG_SIZE, IMG_SIZE),\n"
    "        batch_size=BATCH_SIZE,\n"
    "        label_mode='categorical',\n"
    "        class_names=class_names,\n"
    "        seed=SEED,\n"
    "    )\n"
    "    if subset is not None:\n"
    "        kwargs['validation_split'] = val_split\n"
    "        kwargs['subset'] = subset\n"
    "    kwargs['shuffle'] = (subset == 'training') or (subset is None and augment)\n"
    "    ds = tf.keras.utils.image_dataset_from_directory(**kwargs)\n"
    "\n"
    "    def preprocess(x, y):\n"
    "        return preprocess_input(x), y\n"
    "\n"
    "    def augment_fn(x, y):\n"
    "        x = tf.image.random_flip_left_right(x)\n"
    "        x = tf.image.random_brightness(x, 0.2)\n"
    "        x = tf.image.random_contrast(x, 0.8, 1.2)\n"
    "        return x, y\n"
    "\n"
    "    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)\n"
    "    if augment:\n"
    "        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)\n"
    "    return ds.cache().prefetch(AUTOTUNE)\n"
    "\n"
    "\n"
    "if DATASET_MODE == 'split':\n"
    "    train_ds = make_ds(TRAIN_DIR, augment=True)\n"
    "    val_ds   = make_ds(VAL_DIR,   augment=False)\n"
    "    print('Using pre-split train/val directories.')\n"
    "elif DATASET_MODE == 'flat':\n"
    "    train_ds = make_ds(DATA_ROOT, augment=True,  subset='training',   val_split=VAL_SPLIT)\n"
    "    val_ds   = make_ds(DATA_ROOT, augment=False, subset='validation', val_split=VAL_SPLIT)\n"
    "    print(f'Using flat structure with {int(VAL_SPLIT*100)}% validation split.')\n"
    "\n"
    "print('Datasets ready.')\n"
))

# ── 5. Model ──────────────────────────────────────────────────────────────────
cells.append(md_cell("## 5. Model Architecture"))
cells.append(code_cell(
    "base_model = MobileNetV2(\n"
    "    input_shape=(IMG_SIZE, IMG_SIZE, 3),\n"
    "    include_top=False,\n"
    "    weights='imagenet',\n"
    ")\n"
    "base_model.trainable = False\n"
    "\n"
    "inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))\n"
    "x       = base_model(inputs, training=False)\n"
    "x       = layers.GlobalAveragePooling2D()(x)\n"
    "x       = layers.BatchNormalization()(x)\n"
    "x       = layers.Dense(512, activation='relu')(x)\n"
    "x       = layers.BatchNormalization()(x)\n"
    "x       = layers.Dropout(0.4)(x)\n"
    "x       = layers.Dense(256, activation='relu')(x)\n"
    "x       = layers.Dropout(0.3)(x)\n"
    "outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)\n"
    "\n"
    "model = models.Model(inputs, outputs)\n"
    "model.compile(\n"
    "    optimizer=tf.keras.optimizers.Adam(1e-4),\n"
    "    loss='categorical_crossentropy',\n"
    "    metrics=['accuracy'],\n"
    ")\n"
    "model.summary()\n"
))

# ── 6. Phase 1 Training ───────────────────────────────────────────────────────
cells.append(md_cell("## 6. Phase 1 — Initial Training"))
cells.append(code_cell(
    "cb1 = [\n"
    "    callbacks.ModelCheckpoint(\n"
    "        os.path.join(OUTPUT_DIR, 'mobilenet_arabic_best_initial.h5'),\n"
    "        monitor='val_accuracy', save_best_only=True, verbose=1),\n"
    "    callbacks.EarlyStopping(\n"
    "        monitor='val_accuracy', patience=4,\n"
    "        restore_best_weights=True, verbose=1),\n"
    "    callbacks.ReduceLROnPlateau(\n"
    "        monitor='val_loss', factor=0.5, patience=2, verbose=1),\n"
    "    callbacks.CSVLogger(\n"
    "        os.path.join(OUTPUT_DIR, 'training_initial.csv')),\n"
    "]\n"
    "\n"
    "print('=' * 60)\n"
    "print('PHASE 1: INITIAL TRAINING')\n"
    "print('=' * 60)\n"
    "history1 = model.fit(\n"
    "    train_ds,\n"
    "    epochs=INITIAL_EPOCHS,\n"
    "    validation_data=val_ds,\n"
    "    callbacks=cb1,\n"
    ")\n"
))

# ── 7. Phase 2 Fine-Tuning ────────────────────────────────────────────────────
cells.append(md_cell("## 7. Phase 2 — Fine-Tuning"))
cells.append(code_cell(
    "base_model.trainable = True\n"
    "for layer in base_model.layers[:-40]:\n"
    "    layer.trainable = False\n"
    "for layer in base_model.layers:\n"
    "    if isinstance(layer, layers.BatchNormalization):\n"
    "        layer.trainable = False\n"
    "\n"
    "model.compile(\n"
    "    optimizer=tf.keras.optimizers.Adam(1e-5),\n"
    "    loss='categorical_crossentropy',\n"
    "    metrics=['accuracy'],\n"
    ")\n"
    "\n"
    "cb2 = [\n"
    "    callbacks.ModelCheckpoint(\n"
    "        os.path.join(OUTPUT_DIR, 'mobilenet_arabic_best_finetuned.h5'),\n"
    "        monitor='val_accuracy', save_best_only=True, verbose=1),\n"
    "    callbacks.EarlyStopping(\n"
    "        monitor='val_accuracy', patience=4,\n"
    "        restore_best_weights=True, verbose=1),\n"
    "    callbacks.ReduceLROnPlateau(\n"
    "        monitor='val_loss', factor=0.5, patience=2, verbose=1),\n"
    "    callbacks.CSVLogger(\n"
    "        os.path.join(OUTPUT_DIR, 'training_finetune.csv')),\n"
    "]\n"
    "\n"
    "print('=' * 60)\n"
    "print('PHASE 2: FINE-TUNING')\n"
    "print('=' * 60)\n"
    "history2 = model.fit(\n"
    "    train_ds,\n"
    "    epochs=FINETUNE_EPOCHS,\n"
    "    validation_data=val_ds,\n"
    "    callbacks=cb2,\n"
    ")\n"
    "model.save(os.path.join(OUTPUT_DIR, 'mobilenet_arabic_final.h5'))\n"
    "print('Final model saved.')\n"
))

# ── 8. Training History Plot ──────────────────────────────────────────────────
cells.append(md_cell("## 8. Training History"))
cells.append(code_cell(
    "h1, h2 = history1.history, history2.history\n"
    "acc      = h1['accuracy']     + h2['accuracy']\n"
    "val_acc  = h1['val_accuracy'] + h2['val_accuracy']\n"
    "loss     = h1['loss']         + h2['loss']\n"
    "val_loss = h1['val_loss']     + h2['val_loss']\n"
    "ep       = range(1, len(acc) + 1)\n"
    "split    = INITIAL_EPOCHS\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
    "ax1.plot(ep, acc,     'o-', color='orange', label='Train Acc')\n"
    "ax1.plot(ep, val_acc, 'o-', color='red',    label='Val Acc')\n"
    "ax1.axvline(x=split + 0.5, color='green', linestyle='--', label='Fine-tune')\n"
    "ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)\n"
    "\n"
    "ax2.plot(ep, loss,     'o-', color='blue',   label='Train Loss')\n"
    "ax2.plot(ep, val_loss, 'o-', color='purple', label='Val Loss')\n"
    "ax2.axvline(x=split + 0.5, color='green', linestyle='--', label='Fine-tune')\n"
    "ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)\n"
    "plt.show()\n"
    "print(f'Best Val Accuracy: {max(val_acc):.4f}')\n"
))

# ── 9. TTA Evaluation ─────────────────────────────────────────────────────────
cells.append(md_cell("## 9. TTA Evaluation on Test Set"))
cells.append(code_cell(
    "def tta_augment(image):\n"
    "    return np.array([\n"
    "        image,\n"
    "        ndimage.rotate(image,  5, reshape=False, mode='nearest'),\n"
    "        ndimage.rotate(image, -5, reshape=False, mode='nearest'),\n"
    "        np.clip(image * 1.1, 0, 255),\n"
    "        np.clip(image * 0.9, 0, 255),\n"
    "    ])\n"
    "\n"
    "best_path = os.path.join(OUTPUT_DIR, 'mobilenet_arabic_best_finetuned.h5')\n"
    "best_model = tf.keras.models.load_model(best_path) if os.path.exists(best_path) else model\n"
    "print('Model loaded for inference.')\n"
    "\n"
    "# Use TEST_DIR if available, else sample one image per class from val\n"
    "_test_root = TEST_DIR if (TEST_DIR and os.path.exists(TEST_DIR)) else (\n"
    "    VAL_DIR if (VAL_DIR and os.path.exists(VAL_DIR)) else DATA_ROOT\n"
    ")\n"
    "print('Test root: ' + str(_test_root))\n"
    "\n"
    "test_images, image_names = [], []\n"
    "for cls in sorted(os.listdir(_test_root)):\n"
    "    cls_path = os.path.join(_test_root, cls)\n"
    "    if not os.path.isdir(cls_path) or cls not in class_names:\n"
    "        continue\n"
    "    imgs = [f for f in os.listdir(cls_path)\n"
    "            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n"
    "    if imgs:\n"
    "        img = cv2.imread(os.path.join(cls_path, imgs[0]))\n"
    "        if img is not None:\n"
    "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
    "            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)\n"
    "            test_images.append(img)\n"
    "            image_names.append(cls)\n"
    "\n"
    "test_images = np.array(test_images)\n"
    "all_preds = []\n"
    "for img in test_images:\n"
    "    t = preprocess_input(tta_augment(img).copy())\n"
    "    p = best_model.predict(t, verbose=0, batch_size=5)\n"
    "    all_preds.append(np.mean(p, axis=0))\n"
    "\n"
    "all_preds    = np.array(all_preds)\n"
    "pred_classes = np.argmax(all_preds, axis=1)\n"
    "pred_confs   = np.max(all_preds,   axis=1)\n"
    "\n"
    "correct  = sum(image_names[i] == class_names[pred_classes[i]] for i in range(len(image_names)))\n"
    "accuracy = correct / max(len(image_names), 1) * 100\n"
    "avg_conf = pred_confs.mean() * 100 if len(pred_confs) else 0\n"
    "\n"
    "print(f'Test images: {len(test_images)}')\n"
    "print(f'Test Accuracy (TTA): {accuracy:.2f}%')\n"
    "print(f'Avg Confidence: {avg_conf:.1f}%')\n"
    "print()\n"
    "for i, name in enumerate(image_names):\n"
    "    pred = class_names[pred_classes[i]]\n"
    "    conf = pred_confs[i] * 100\n"
    "    mark = 'OK' if pred == name else 'XX'\n"
    "    print(f'  [{mark}] {name:10s} -> {pred:10s} ({conf:.1f}%)')\n"
))

# ── 10. Confusion Matrix ──────────────────────────────────────────────────────
cells.append(md_cell("## 10. Confusion Matrix"))
cells.append(code_cell(
    "pred_labels = [class_names[pred_classes[i]] for i in range(len(image_names))]\n"
    "label_set   = sorted(set(class_names))\n"
    "cm = confusion_matrix(list(image_names), pred_labels, labels=label_set)\n"
    "\n"
    "plt.figure(figsize=(14, 12))\n"
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
    "            xticklabels=label_set, yticklabels=label_set)\n"
    "plt.xlabel('Predicted'); plt.ylabel('True')\n"
    "plt.title('Confusion Matrix — Test Set (TTA)', fontweight='bold')\n"
    "plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)\n"
    "plt.show()\n"
))

# ── 11. Summary ───────────────────────────────────────────────────────────────
cells.append(md_cell("## 11. Final Summary"))
cells.append(code_cell(
    "print('=' * 60)\n"
    "print('ARSL MOBILENETV2 KAGGLE EDITION - SUMMARY')\n"
    "print('=' * 60)\n"
    "print(f'  Mode:            {DATASET_MODE}')\n"
    "print(f'  Classes:         {NUM_CLASSES}')\n"
    "print(f'  Image size:      {IMG_SIZE}x{IMG_SIZE}')\n"
    "print(f'  Batch size:      {BATCH_SIZE}')\n"
    "print(f'  Initial epochs:  {INITIAL_EPOCHS}')\n"
    "print(f'  Finetune epochs: {FINETUNE_EPOCHS}')\n"
    "print(f'  Test Accuracy:   {accuracy:.2f}%')\n"
    "print()\n"
    "for fname in [\n"
    "    'mobilenet_arabic_best_initial.h5',\n"
    "    'mobilenet_arabic_best_finetuned.h5',\n"
    "    'mobilenet_arabic_final.h5',\n"
    "    'training_initial.csv',\n"
    "    'training_finetune.csv',\n"
    "    'training_history.png',\n"
    "    'confusion_matrix.png',\n"
    "]:\n"
    "    path = os.path.join(OUTPUT_DIR, fname)\n"
    "    mark = 'OK' if os.path.exists(path) else '--'\n"
    "    print(f'  [{mark}] {fname}')\n"
))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

output_path = "ArSL_MobileNetV2_Kaggle.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {output_path}")
print(f"Cells:   {len(cells)}")

"""
============================================================
UNIFIED MERGE + RETRAIN PIPELINE
============================================================

This script:
1. Merges CSV datasets for ASL Letters (2 CSVs) and ArSL Letters (1 CSV)
2. Retrains the English ASL MLP model on merged data
3. Retrains the Arabic ArSL MLP model on merged data

Run with: Python 3.10.11
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ============================================================
# GPU CONFIGURATION
# ============================================================
print("=" * 70)
print("GPU DETECTION AND CONFIGURATION")
print("=" * 70)
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
USE_GPU = False
DEVICE = '/CPU:0'

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        USE_GPU = True
        DEVICE = '/GPU:0'
        print(f"[OK] GPU configured: {gpus[0]}")
        
        # Mixed precision
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[OK] Mixed precision enabled: {policy.name}")
        except Exception as e:
            print(f"[WARN] Mixed precision not enabled: {e}")
    except RuntimeError as e:
        print(f"[WARN] GPU config error: {e}")
else:
    print("[INFO] No GPU found - using CPU")

print(f"[OK] Using device: {DEVICE}")
print("=" * 70)

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path('.').resolve()
print(f"\nProject Root: {PROJECT_ROOT}")

# ============================================================
# STEP 1: MERGE DATASETS
# ============================================================

def normalize_csv_columns(df, expected_features=63):
    """
    Normalize CSV column names to a standard format.
    
    The two ASL CSV files have DIFFERENT column names:
    - Dataset 1: 0, 1, 2, ..., 62, label  (numeric)
    - Dataset 2: x0, y0, z0, ..., x20, y20, z20, label  (named)
    
    We standardize all to: x0, y0, z0, ..., x20, y20, z20, label
    """
    # Find the label column
    if 'label' in df.columns:
        label_col = 'label'
    elif 'letter' in df.columns:
        label_col = 'letter'
    else:
        raise ValueError("No 'label' or 'letter' column found!")
    
    # Extract labels
    labels = df[label_col].values
    
    # Get feature columns (everything except label)
    feature_cols = [c for c in df.columns if c != label_col]
    
    if len(feature_cols) != expected_features:
        print(f"   [WARN] Expected {expected_features} features, got {len(feature_cols)}")
    
    # Extract features as numeric array
    features = df[feature_cols].astype('float32').values
    
    # Create standardized column names
    std_cols = []
    for i in range(21):  # 21 hand landmarks
        std_cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    
    # Build new DataFrame with standard column names
    df_std = pd.DataFrame(features, columns=std_cols[:features.shape[1]])
    df_std['label'] = labels
    
    return df_std


def load_and_normalize_csv(source_config, expected_features=63):
    """Load a CSV dataset and normalize its column names."""
    path = source_config['path']
    
    if not path.exists():
        if source_config.get('required', True):
            raise FileNotFoundError(f"[ERROR] REQUIRED dataset not found: {path}")
        else:
            print(f"[WARN] Optional dataset not found: {path}")
            return None
    
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"   Loading: {path.name} ({size_mb:.1f} MB)")
    df = pd.read_csv(path)
    
    # Normalize columns
    df = normalize_csv_columns(df, expected_features)
    
    print(f"      [OK] Loaded {len(df):,} rows, {df['label'].nunique()} classes, {len(df.columns)-1} features")
    return df


def balance_classes(df, max_samples_per_class, label_col='label'):
    """Balance dataset by capping each class at max_samples_per_class."""
    print(f"\n   BALANCING CLASSES (max {max_samples_per_class} per class)")

    before = len(df)
    balanced_dfs = []
    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        if len(cls_df) > max_samples_per_class:
            cls_df = cls_df.sample(n=max_samples_per_class, random_state=42)
        balanced_dfs.append(cls_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    after = len(df_balanced)
    removed = before - after

    print(f"      Before: {before:,} samples")
    print(f"      After:  {after:,} samples")
    print(f"      Removed: {removed:,} (excess from over-represented classes)")

    return df_balanced


def merge_letter_dataset(model_key, config):
    """Merge letter datasets (CSV format) with column normalization."""

    print(f"\n{'='*70}")
    print(f"MERGING: {config['name']}")
    print(f"{'='*70}")

    all_dfs = []

    # Step 1: Load all sources
    print(f"\nSTEP 1: LOADING DATA SOURCES")
    for source in config['data_sources']:
        print(f"\n   Source: {source['name']}")
        df = load_and_normalize_csv(source)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print(f"[ERROR] No datasets loaded for {model_key}")
        return None

    # Step 2: Merge (now all DFs have the same columns!)
    print(f"\nSTEP 2: MERGING ALL SOURCES")
    df_merged = pd.concat(all_dfs, ignore_index=True)
    print(f"   Combined: {len(df_merged):,} samples, {df_merged['label'].nunique()} classes, {len(df_merged.columns)-1} features")

    # Step 3: Balance classes
    print(f"\nSTEP 3: CLASS BALANCING")
    df_balanced = balance_classes(df_merged, config['max_samples_per_class'])

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   [OK] Shuffled")

    # Step 4: Save
    print(f"\nSTEP 4: SAVING MERGED DATASET")
    config['output'].parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(config['output'], index=False)
    print(f"   [OK] Saved: {config['output']}")
    print(f"   Total: {len(df_balanced):,} samples, {df_balanced['label'].nunique()} classes, {len(df_balanced.columns)-1} features")

    # Statistics
    stats = {
        'total_samples': len(df_balanced),
        'num_classes': int(df_balanced['label'].nunique()),
        'num_features': len(df_balanced.columns) - 1,
        'class_distribution': df_balanced['label'].value_counts().to_dict(),
        'samples_per_class': {
            'min': int(df_balanced.groupby('label').size().min()),
            'max': int(df_balanced.groupby('label').size().max()),
            'mean': float(df_balanced.groupby('label').size().mean()),
        },
    }
    stats_path = config['output'].parent / f"{config['output'].stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"   [OK] Statistics: {stats_path}")

    print(f"\n[OK] {model_key.upper()} MERGE COMPLETE")
    return df_balanced


# ---- Define merge configs ----
MERGE_CONFIG = {
    'asl_letters': {
        'name': 'ASL Letters (English)',
        'data_sources': [
            {
                'name': 'Kaggle ASL Alphabet 1 (59k samples)',
                'path': PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_mediapipe_keypoints_dataset.csv',
                'label_column': 'label',
                'required': True,
            },
            {
                'name': 'Kaggle ASL Alphabet 2 (2k samples)',
                'path': PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_landmarks_final.csv',
                'label_column': 'label',
                'required': True,
            },
        ],
        'output': PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_letters_merged.csv',
        'max_samples_per_class': 3000,
    },
    'arsl_letters': {
        'name': 'ArSL Letters (Arabic)',
        'data_sources': [
            {
                'name': 'Arabic Sign Language Letters Dataset (8k samples)',
                'path': PROJECT_ROOT / 'ArSL (Arabic Letters)/FINAL_CLEAN_DATASET.csv',
                'label_column': 'label',
                'required': True,
            },
        ],
        'output': PROJECT_ROOT / 'ArSL (Arabic Letters)/arsl_letters_merged.csv',
        'max_samples_per_class': 3000,
    },
}


# ---- Run merges ----
print(f"\n{'='*70}")
print("STARTING DATASET MERGING PIPELINE")
print(f"{'='*70}")

merge_letter_dataset('asl_letters', MERGE_CONFIG['asl_letters'])
merge_letter_dataset('arsl_letters', MERGE_CONFIG['arsl_letters'])

print(f"\n{'='*70}")
print("[OK] ALL MERGES COMPLETE")
print(f"{'='*70}")


# ============================================================
# STEP 2: TRAIN MODELS ON MERGED DATA
# ============================================================

def train_model(csv_path, model_save_path, best_model_save_path, model_name):
    """
    Train an MLP model on a merged CSV dataset.
    
    The CSV must have standardized columns: x0, y0, z0, ..., x20, y20, z20, label
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return None

    # Load dataset
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Samples: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Find label column
    if 'label' in df.columns:
        label_col = 'label'
    elif 'letter' in df.columns:
        df = df.rename(columns={'letter': 'label'})
        label_col = 'label'
    else:
        print("[ERROR] No 'label' or 'letter' column found!")
        return None

    # Separate features and labels
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].astype('float32').values
    y = df[label_col].values

    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    print(f"\nClasses found ({num_classes}):")
    for i, cls in enumerate(encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"   {cls}: {count} samples")

    # Split: 80% train, 20% test; then 80/20 train/val
    print(f"\nSplitting data...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # One-hot encode
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")

    # Build tf.data pipeline
    BATCH_SIZE = 256 if USE_GPU else 64
    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(features, labels, batch_size, training=True):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if training:
            buffer = min(len(features), 10000)
            ds = ds.shuffle(buffer_size=buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train, y_train_cat, BATCH_SIZE, training=True)
    val_ds = make_dataset(X_val, y_val_cat, BATCH_SIZE, training=False)
    test_ds = make_dataset(X_test, y_test_cat, BATCH_SIZE, training=False)

    # Build model
    print(f"\nBuilding MLP model...")
    tf.keras.backend.clear_session()

    with tf.device(DEVICE):
        model = Sequential([
            Dense(
                512,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                input_shape=(X_train.shape[1],),
                name='dense_512'
            ),
            BatchNormalization(name='bn_1'),
            Dropout(0.3, name='dropout_1'),
            Dense(
                256,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='dense_256'
            ),
            BatchNormalization(name='bn_2'),
            Dropout(0.25, name='dropout_2'),
            Dense(
                64,
                activation='relu',
                kernel_initializer='he_normal',
                name='dense_64'
            ),
            Dropout(0.2, name='dropout_3'),
            Dense(num_classes, activation='softmax', dtype='float32', name='output')
        ])

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    print("\nModel Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(best_model_save_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    print(f"\nStarting training on {DEVICE}...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max epochs: 100 (with early stopping)")
    print("=" * 70)

    start_time = time.time()
    with tf.device(DEVICE):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=callbacks,
            verbose=1
        )
    training_time = time.time() - start_time

    # Save final model
    model.save(str(model_save_path))

    print(f"\n{'='*70}")
    print(f"[OK] TRAINING COMPLETE: {model_name}")
    print(f"{'='*70}")
    print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Best model: {best_model_save_path}")
    print(f"Final model: {model_save_path}")

    if hasattr(history, 'history'):
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nFinal Metrics:")
        print(f"   Training Accuracy:   {final_train_acc*100:.2f}%")
        print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")

    # Evaluate on test set
    print(f"\nEvaluating on test set ({len(X_test):,} samples)...")
    with tf.device(DEVICE):
        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 70)

    return history


# ---- Train ASL English Model ----
asl_merged_csv = PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_letters_merged.csv'
asl_model_path = PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_mediapipe_mlp_model.h5'
asl_best_model_path = PROJECT_ROOT / 'Base_Pipeline_English_Letters/asl_mediapipe_mlp_model_best.h5'

if asl_merged_csv.exists():
    train_model(
        csv_path=str(asl_merged_csv),
        model_save_path=str(asl_model_path),
        best_model_save_path=str(asl_best_model_path),
        model_name="ASL English Letters (Merged)"
    )
else:
    print(f"[WARN] Skipping ASL training - merged CSV not found: {asl_merged_csv}")


# ---- Train ArSL Arabic Model ----
arsl_merged_csv = PROJECT_ROOT / 'ArSL (Arabic Letters)/arsl_letters_merged.csv'
arsl_model_path = PROJECT_ROOT / 'ArSL (Arabic Letters)/arsl_mediapipe_mlp_model_final.h5'
arsl_best_model_path = PROJECT_ROOT / 'ArSL (Arabic Letters)/arsl_mediapipe_mlp_model_best.h5'

if arsl_merged_csv.exists():
    train_model(
        csv_path=str(arsl_merged_csv),
        model_save_path=str(arsl_model_path),
        best_model_save_path=str(arsl_best_model_path),
        model_name="ArSL Arabic Letters (Merged)"
    )
else:
    print(f"[WARN] Skipping ArSL training - merged CSV not found: {arsl_merged_csv}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("ALL DONE - MERGE + RETRAIN PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"\nOutput files:")
print(f"   ASL Merged CSV:    {asl_merged_csv}")
print(f"   ASL Model (.h5):   {asl_model_path}")
print(f"   ArSL Merged CSV:   {arsl_merged_csv}")
print(f"   ArSL Model (.h5):  {arsl_model_path}")
print(f"\nNext: Run Production_Architecture_English.ipynb or")
print(f"   Production_Architecture_Arabic.ipynb to test predictions!")
print(f"{'='*70}")

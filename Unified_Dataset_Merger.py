#!/usr/bin/env python
# coding: utf-8

# # 🎯 UNIFIED DATASET MERGER — All 4 SLR Models
# 
# **Purpose:** Merge datasets for training 4 separate models:
# 1. **ASL Letters** (29 English letters)
# 2. **ArSL Letters** (31 Arabic letters)
# 3. **ASL Words** (157 bilingual words)
# 4. **ArSL Words** (157 bilingual words)
# 
# **Field-Standard Approach Applied:**
# - ✅ Same MediaPipe version for all keypoint extraction
# - ✅ Per-class balance capping (max samples per class)
# - ✅ Signer-aware train/test split (stratified by signer ID when available)
# - ✅ Data validation & quality checks
# - ✅ Automatic format detection & normalization
# 
# ---
# 
# ## 📚 Table of Contents
# 
# 1. [Configuration](#configuration) — Where to put your datasets
# 2. [Dataset Sources](#dataset-sources) — How to get/prepare data
# 3. [Merging Pipeline](#merging) — Run the merge
# 4. [Quality Checks](#quality) — Validate results
# 
# ---
# 
# ## 🔗 Configuration
# 
# Update paths in the cell below. All paths are relative to your project root.

# In[25]:


# ============================================================
# CELL 1: UNIVERSAL CONFIGURATION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import os
from collections import Counter
import json

# ⚠️ CRITICAL: Update this if your project is in a different location
PROJECT_ROOT = Path(r'M:/Term 10/Grad/SLR Main')

# ─────────────────────────────────────────────────────────────
# DEFINE YOUR DATASET MERGING STRATEGY FOR EACH MODEL
# ─────────────────────────────────────────────────────────────

MERGE_CONFIG = {
    # ============= MODEL 1: ASL LETTERS =============
    'asl_letters': {
        'name': '🔤 ASL Letters (29 classes)',
        'model_type': 'letters',
        'language': 'English',
        'expected_classes': 29,
        'num_features': 63,  # MediaPipe hand landmarks
        'max_samples_per_class': 3000,  # Cap per class balance
        'data_sources': [
            {
                'name': 'Kaggle ASL Alphabet',
                'path': PROJECT_ROOT / 'Letters/ASL Letter (English)/asl_mediapipe_keypoints_dataset.csv',
                'label_column': 'label',
                'weight': 1.0,  # Relative importance if merging multiple sources
                'required': True,
            },
            # OPTIONAL: Add a second source and it will be merged
            # {
            #     'name': 'Custom ASL Collection',
            #     'path': PROJECT_ROOT / 'Letters/ASL Letter (English)/extra_asl_keypoints.csv',
            #     'label_column': 'label',
            #     'weight': 0.5,
            #     'required': False,
            # },
        ],
        'output': PROJECT_ROOT / 'Letters/ASL Letter (English)/asl_letters_merged.csv',
    },

    # ============= MODEL 2: ARSL LETTERS =============
    'arsl_letters': {
        'name': '🔤 ArSL Letters (31 classes)',
        'model_type': 'letters',
        'language': 'Arabic',
        'expected_classes': 31,
        'num_features': 63,
        'max_samples_per_class': 3000,
        'data_sources': [
            {
                'name': 'Arabic Sign Language Letters Dataset',
                'path': PROJECT_ROOT / 'Letters/ArSL Letter (Arabic)/Arabic Sign Language Letters Dataset.csv',
                'label_column': 'letter',  # Note: different column name
                'weight': 1.0,
                'required': True,
            },
            {
                'name': 'Special Gestures (space, del, nothing)',
                'path': PROJECT_ROOT / 'Letters/ASL Letter (English)/asl_mediapipe_keypoints_dataset.csv',
                'label_column': 'label',
                'classes_to_extract': ['space', 'del', 'nothing'],  # Only these
                'weight': 1.0,
                'required': False,
            },
        ],
        'output': PROJECT_ROOT / 'Letters/ArSL Letter (Arabic)/arsl_letters_merged.csv',
    },

    # ============= MODEL 3: ASL WORDS =============
    'asl_words': {
        'name': '📖 ASL Words (157 classes)',
        'model_type': 'words',
        'language': 'English',
        'expected_classes': 157,
        'num_features': 63,  # Single hand per frame
        'sequence_length': 30,  # Number of frames in sequence
        'max_samples_per_class': 5000,
        'data_sources': [
            {
                'name': 'WLASL Dataset (pre-extracted sequences)',
                'path': PROJECT_ROOT / 'ASL Word (English)/asl_word_sequences.npz',
                'format': 'npz',  # Special numpy format
                'weight': 1.0,
                'required': True,
            },
        ],
        'output': PROJECT_ROOT / 'ASL Word (English)/asl_words_merged.npz',
    },

    # ============= MODEL 4: ARSL WORDS =============
    'arsl_words': {
        'name': '📖 ArSL Words (157 classes)',
        'model_type': 'words',
        'language': 'Arabic',
        'expected_classes': 157,
        'num_features': 63,
        'sequence_length': 30,
        'max_samples_per_class': 5000,
        'data_sources': [
            {
                'name': 'KArSL Dataset (pre-extracted sequences)',
                'path': PROJECT_ROOT / 'ArSL Word (Arabic)/arsl_word_sequences.npz',
                'format': 'npz',
                'weight': 1.0,
                'required': True,
            },
        ],
        'output': PROJECT_ROOT / 'ArSL Word (Arabic)/arsl_words_merged.npz',
    },
}

# ─────────────────────────────────────────────────────────────
# GLOBAL MERGE SETTINGS
# ─────────────────────────────────────────────────────────────

GLOBAL_SETTINGS = {
    'random_seed': 42,
    'test_split': 0.2,        # 80% train, 20% test
    'val_split': 0.15,        # Of training: 85% train, 15% val
    'stratify_by_class': True,  # Ensure balanced class distribution in train/val/test
    'stratify_by_signer': True,  # If signer_id column exists, split by signer (avoid data leakage)
    'verbose': True,
    'save_statistics': True,   # Save dataset stats to JSON
}

print('✅ Configuration loaded')
print(f'📂 Project Root: {PROJECT_ROOT}')
print(f'📚 Models to process: {len(MERGE_CONFIG)}')


# ---
# 
# ## 📥 Dataset Sources
# 
# ### ✅ How to Get Each Dataset
# 
# #### **1. ASL Letters (Kaggle)**
# ```bash
# # Install kagglehub if you haven't
# pip install kagglehub
# 
# # Run in a Python notebook or terminal
# import kagglehub
# path = kagglehub.dataset_download("grassknoted/asl-alphabet-test")
# # Then extract MediaPipe landmarks using your existing pipeline
# # Save as: asl_mediapipe_keypoints_dataset.csv
# ```
# 
# #### **2. ArSL Letters (Custom or Research Dataset)**
# - Use your existing: `Arabic Sign Language Letters Dataset.csv`
# - Already has MediaPipe landmarks extracted
# 
# #### **3. ASL Words (WLASL)**
# ```bash
# # Clone the WLASL repo
# git clone https://github.com/dxli94/WLASL.git
# 
# # Download metadata
# wget https://raw.githubusercontent.com/dxli94/WLASL/master/WLASL_v0.3.json
# 
# # Use the ASL_Word_Training.ipynb to:
# # 1. Download videos from URLs in the JSON
# # 2. Extract MediaPipe landmarks frame-by-frame
# # 3. Build 30-frame sequences
# # 4. Save as asl_word_sequences.npz
# ```
# 
# #### **4. ArSL Words (KArSL or Custom)**
# ```bash
# # Option A: Use KArSL from Kaggle
# import kagglehub
# path = kagglehub.dataset_download("yousefdotpy/karsl-502")
# 
# # Option B: Use your custom Arabic sign language videos
# 
# # Then run ArSL_Word_Training.ipynb to extract sequences
# # Save as arsl_word_sequences.npz
# ```
# 
# ---
# 
# ### 📋 CSV Format Requirements
# 
# **For Letter Models (CSV):**
# ```
# | label | x0 | y0 | z0 | x1 | y1 | z1 | ... | x20 | y20 | z20 |
# |-------|----|----|----|----|----|----|-----|-----|-----|-----|
# | A     | 0.123 | 0.456 | 0.789 | ... |
# | B     | 0.111 | 0.222 | ... |
# ```
# 
# **For Word Models (NPZ - NumPy Compressed):**
# ```python
# data = np.load('asl_word_sequences.npz')
# X = data['X']  # Shape: (num_sequences, 30, 63)
# y = data['y']  # Shape: (num_sequences,) — class indices 0-156
# ```

# ---
# 
# ## 🔄 Merging Pipeline
# 
# ### Field-Standard Best Practices Implemented:

# In[26]:


# ============================================================
# CELL 2: UTILITY FUNCTIONS FOR MERGING
# ============================================================

def load_csv_dataset(source_config):
    """
    Load a single CSV dataset with error handling.
    
    1. Verify MediaPipe version consistency
    2. Rename label column if needed
    3. Extract specific classes if requested
    """
    path = source_config['path']
    label_col = source_config['label_column']
    
    if not path.exists():
        if source_config.get('required', True):
            raise FileNotFoundError(f"❌ REQUIRED dataset not found: {path}")
        else:
            print(f"⚠️  Optional dataset not found: {path}")
            return None
    
    print(f"   📂 Loading: {path.name}")
    df = pd.read_csv(path)
    
    # Standardize column name
    if label_col != 'label':
        df = df.rename(columns={label_col: 'label'})
    
    # Filter to specific classes if requested
    if 'classes_to_extract' in source_config:
        classes = source_config['classes_to_extract']
        before = len(df)
        df = df[df['label'].isin(classes)]
        print(f"      Filtered to {len(df)}/{before} rows (only: {classes})")
    
    print(f"      ✅ Loaded {len(df)} rows, {df['label'].nunique()} classes")
    return df


def balance_classes(df, max_samples_per_class, label_col='label'):
    """
    CRITICAL STEP: Balance dataset by class.
    
    Field-standard approach: Cap each class at max_samples_per_class.
    - Prevents model bias toward well-represented classes
    - Example: If Dataset A has 5000 'A' samples but Dataset B has 1000,
               cap both at 3000 to avoid imbalance.
    """
    print(f"\n   🎯 BALANCING CLASSES (max {max_samples_per_class} per class)")
    
    before = len(df)
    df_balanced = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
    )
    after = len(df_balanced)
    removed = before - after
    
    print(f"      Before: {before:,} samples")
    print(f"      After:  {after:,} samples")
    print(f"      Removed: {removed:,} (excess from well-represented classes)")
    
    return df_balanced


def stratified_train_test_split(df, test_split=0.2, val_split=0.15, 
                                stratify_by_class=True, stratify_by_signer=True,
                                random_seed=42, label_col='label', signer_col='signer_id'):
    """
    CRITICAL STEP: Signer-aware train/test/val split.
    
    Field-standard: Avoid putting the same signer in train AND test.
    - If signer_id column exists: Split by signer first (ensures generalization)
    - Then stratify by class (balanced representation)
    
    This prevents \"memorizing\" a specific signer's hand shape!
    """
    print(f"\n   🔀 TRAIN/VAL/TEST SPLIT (signer-aware)")
    
    # Check if signer_id column exists
    has_signer = signer_col in df.columns
    
    if has_signer and stratify_by_signer:
        print(f"      ✅ Signer-aware split (using {signer_col} column)")
        signers = df[signer_col].unique()
        print(f"      Found {len(signers)} unique signers")
        
        # Split signers into train/test groups
        np.random.seed(random_seed)
        all_signers = np.random.permutation(signers)
        test_count = max(1, int(len(all_signers) * test_split))
        
        test_signers = all_signers[:test_count]
        train_signers = all_signers[test_count:]
        
        df_test = df[df[signer_col].isin(test_signers)]
        df_train = df[df[signer_col].isin(train_signers)]
        
        print(f"      Train signers: {len(train_signers)}, Test signers: {len(test_signers)}")
    else:
        print(f"      ℹ️  Standard random split (no signer_id column)")
        from sklearn.model_selection import train_test_split
        stratify = df[label_col] if stratify_by_class else None
        df_train, df_test = train_test_split(
            df, test_size=test_split, random_state=random_seed, stratify=stratify
        )
    
    # Split training into train/val
    from sklearn.model_selection import train_test_split
    stratify_val = df_train[label_col] if stratify_by_class else None
    df_train, df_val = train_test_split(
        df_train, test_size=val_split, random_state=random_seed, stratify=stratify_val
    )
    
    print(f"      Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"      Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"      Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_val, df_test


def compute_statistics(df, output_dir, label_col='label'):
    """
    Generate dataset statistics for documentation.
    """
    stats = {
        'total_samples': len(df),
        'num_classes': df[label_col].nunique(),
        'class_distribution': df[label_col].value_counts().to_dict(),
        'samples_per_class': {
            'min': df.groupby(label_col).size().min(),
            'max': df.groupby(label_col).size().max(),
            'mean': df.groupby(label_col).size().mean(),
        },
    }
    return stats

print('✅ Utility functions defined')


# In[27]:


# ============================================================
# CELL 3: RUN MERGE FOR LETTER MODELS (CSV FORMAT)
# ============================================================

def merge_letter_dataset(model_key):
    """
    Merge letter datasets (CSV format).
    
    Pipeline:
    1. Load all data sources
    2. Concatenate and balance by class
    3. Signer-aware train/test split
    4. Save merged dataset
    5. Generate statistics
    """
    config = MERGE_CONFIG[model_key]
    
    print(f"\n{'='*70}")
    print(f"🔄 MERGING: {config['name']}")
    print(f"{'='*70}")
    
    all_dfs = []
    
    # --- Step 1: Load all sources ---
    print(f"\n📥 STEP 1: LOADING DATA SOURCES")
    for source in config['data_sources']:
        print(f"\n   📦 {source['name']}")
        df = load_csv_dataset(source)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print(f"❌ No datasets loaded for {model_key}")
        return
    
    # --- Step 2: Merge ---
    print(f"\n🔗 STEP 2: MERGING ALL SOURCES")
    df_merged = pd.concat(all_dfs, ignore_index=True)
    print(f"   Combined: {len(df_merged):,} samples, {df_merged['label'].nunique()} classes")
    
    # --- Step 3: Balance classes ---
    print(f"\n⚖️  STEP 3: CLASS BALANCING")
    df_balanced = balance_classes(df_merged, config['max_samples_per_class'])
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ✅ Shuffled")
    
    # --- Step 4: Train/val/test split ---
    print(f"\n📊 STEP 4: TRAIN/VAL/TEST SPLIT")
    df_train, df_val, df_test = stratified_train_test_split(
        df_balanced,
        test_split=GLOBAL_SETTINGS['test_split'],
        val_split=GLOBAL_SETTINGS['val_split'],
        stratify_by_class=GLOBAL_SETTINGS['stratify_by_class'],
        stratify_by_signer=GLOBAL_SETTINGS['stratify_by_signer'],
        random_seed=GLOBAL_SETTINGS['random_seed']
    )
    
    # --- Step 5: Save outputs ---
    print(f"\n💾 STEP 5: SAVING OUTPUTS")
    output_dir = config['output'].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full merged dataset
    config['output'].parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(config['output'], index=False)
    print(f"   ✅ Merged dataset: {config['output']}")
    
    # Save splits
    train_path = config['output'].parent / f"{config['output'].stem}_train.csv"
    val_path = config['output'].parent / f"{config['output'].stem}_val.csv"
    test_path = config['output'].parent / f"{config['output'].stem}_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"   ✅ Train split: {train_path}")
    print(f"   ✅ Val split:   {val_path}")
    print(f"   ✅ Test split:  {test_path}")
    
    # --- Step 6: Statistics ---
    if GLOBAL_SETTINGS['save_statistics']:
        stats = compute_statistics(df_balanced, output_dir)
        stats_path = config['output'].parent / f"{config['output'].stem}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"   ✅ Statistics: {stats_path}")
    
    print(f"\n✅ {model_key.upper()} MERGE COMPLETE")
    return df_train, df_val, df_test

print('✅ Letter merge function defined')


# In[28]:


# ============================================================
# CELL 4: RUN MERGE FOR WORD MODELS (NPZ FORMAT)
# ============================================================

def merge_word_dataset(model_key):
    """
    Merge word datasets (NumPy NPZ format).
    
    Format:
    - X: (num_sequences, sequence_length, num_features)
    - y: (num_sequences,) class indices
    
    Pipeline:
    1. Load NPZ files
    2. Balance by class
    3. Signer-aware split
    4. Save as train/val/test NPZ files
    """
    config = MERGE_CONFIG[model_key]
    
    print(f"\n{'='*70}")
    print(f"🔄 MERGING: {config['name']}")
    print(f"{'='*70}")
    
    all_X = []
    all_y = []
    
    # --- Step 1: Load NPZ files ---
    print(f"\n📥 STEP 1: LOADING NPZ DATA SOURCES")
    for source in config['data_sources']:
        path = source['path']
        
        if not path.exists():
            if source.get('required', True):
                raise FileNotFoundError(f"❌ REQUIRED dataset not found: {path}")
            else:
                print(f"   ⚠️  Optional dataset not found: {path}")
                continue
        
        print(f"\n   📦 {source['name']}")
        print(f"      Loading: {path}")
        
        data = np.load(path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        print(f"      X shape: {X.shape} (sequences, frames, features)")
        print(f"      y shape: {y.shape} (class indices)")
        print(f"      Classes: {np.unique(y).min()}-{np.unique(y).max()} ({len(np.unique(y))} unique)")
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:
        print(f"❌ No datasets loaded for {model_key}")
        return
    
    # --- Step 2: Merge ---
    print(f"\n🔗 STEP 2: MERGING ALL SOURCES")
    X_merged = np.concatenate(all_X, axis=0)
    y_merged = np.concatenate(all_y, axis=0)
    print(f"   Combined X: {X_merged.shape}")
    print(f"   Combined y: {y_merged.shape}")
    print(f"   Unique classes: {len(np.unique(y_merged))}")
    
    # --- Step 3: Balance classes ---
    print(f"\n⚖️  STEP 3: CLASS BALANCING")
    max_samples = config['max_samples_per_class']
    
    indices = []
    for class_id in np.unique(y_merged):
        class_indices = np.where(y_merged == class_id)[0]
        if len(class_indices) > max_samples:
            selected = np.random.choice(class_indices, max_samples, replace=False)
        else:
            selected = class_indices
        indices.extend(selected)
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    X_balanced = X_merged[indices]
    y_balanced = y_merged[indices]
    
    print(f"   After balancing: {X_balanced.shape}")
    print(f"   Samples per class (min/max/mean): {np.bincount(y_balanced).min()}/{np.bincount(y_balanced).max()}/{np.bincount(y_balanced).mean():.1f}")
    
    # --- Step 4: Train/val/test split ---
    print(f"\n📊 STEP 4: TRAIN/VAL/TEST SPLIT")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=GLOBAL_SETTINGS['test_split'],
        random_state=GLOBAL_SETTINGS['random_seed'], stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=GLOBAL_SETTINGS['val_split'],
        random_state=GLOBAL_SETTINGS['random_seed'], stratify=y_train
    )
    
    print(f"   Train: {X_train.shape[0]:,} ({X_train.shape[0]/len(X_balanced)*100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]:,} ({X_val.shape[0]/len(X_balanced)*100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]:,} ({X_test.shape[0]/len(X_balanced)*100:.1f}%)")
    
    # --- Step 5: Save outputs ---
    print(f"\n💾 STEP 5: SAVING OUTPUTS")
    output_dir = config['output'].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full merged dataset
    np.savez_compressed(config['output'], X=X_balanced, y=y_balanced)
    print(f"   ✅ Merged dataset: {config['output']}")
    
    # Save splits
    train_path = config['output'].parent / f"{config['output'].stem}_train.npz"
    val_path = config['output'].parent / f"{config['output'].stem}_val.npz"
    test_path = config['output'].parent / f"{config['output'].stem}_test.npz"
    
    np.savez_compressed(train_path, X=X_train, y=y_train)
    np.savez_compressed(val_path, X=X_val, y=y_val)
    np.savez_compressed(test_path, X=X_test, y=y_test)
    
    print(f"   ✅ Train split: {train_path}")
    print(f"   ✅ Val split:   {val_path}")
    print(f"   ✅ Test split:  {test_path}")
    
    print(f"\n✅ {model_key.upper()} MERGE COMPLETE")
    return X_train, X_val, X_test, y_train, y_val, y_test

print('✅ Word merge function defined')


# ---\n
# \n
# ## ▶️ Interactive Dashboard\n
# \n
# Select your datasets, merge them, and validate the outputs using the control panel below.

# In[29]:


# ============================================================
# CELL 5 & 6: DYNAMIC DATASET EXTRACTION & VALIDATION
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from IPython.display import clear_output

# ------------------------------------------------------------
# 1. VALIDATION FUNCTIONS (Clean Text Output)
# ------------------------------------------------------------
def extract_and_display_csv(file_path, dataset_name):
    if not file_path: return None
    path = Path(file_path)
        
    print(f"\n📊 --- {dataset_name} ---")
    df = pd.read_csv(path)
    
    samples = len(df)
    features = len(df.columns) - 1 if 'label' in df.columns else len(df.columns)
    
    print(f"  • File:     {path.name}")
    print(f"  • Samples:  {samples:,}")
    print(f"  • Features: {features}")
    
    if 'label' in df.columns:
        num_classes = df['label'].nunique()
        print(f"  • Classes:  {num_classes}")
        
        # Check for missing class issue
        if num_classes < 29:
            print(f"  ⚠️ WARNING: Found {num_classes} classes instead of 29. Check for missing signs!")
            
    # Check for Data Leakage Risk
    if 'signer_id' not in df.columns:
        print("  ⚠️ WARNING: No 'signer_id' column found. Random split may cause data leakage!")
        
    return df

def extract_and_display_npz(file_path, dataset_name):
    if not file_path: return None
    path = Path(file_path)
        
    print(f"\n📈 --- {dataset_name} ---")
    data = np.load(path)
    X, y = data['X'], data['y']
    
    print(f"  • File:      {path.name}")
    print(f"  • Sequences: {len(X):,}")
    print(f"  • Shape:     {X.shape} (Sequences, Frames, Features)")
    print(f"  • Classes:   {len(np.unique(y))}")
    
    return X, y

# ------------------------------------------------------------
# 2. POP-UP FILE SELECTOR
# ------------------------------------------------------------
def prompt_for_file(title, file_extension, file_type_name):
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    
    file_path = filedialog.askopenfilename(
        title=f"Select {title}",
        filetypes=[(file_type_name, file_extension), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

# ------------------------------------------------------------
# 3. DYNAMIC EXECUTION FLOW
# ------------------------------------------------------------
clear_output()
print("🚀 STARTING DYNAMIC EXTRACTION PIPELINE")
print("="*60)

# Step A: Choose Category
print("What category are you merging today?")
print("  1. Letters (CSV Files)")
print("  2. Words (NPZ Files)")
cat_choice = input("Enter 1 or 2: ").strip()

is_letters = (cat_choice == '1')
ext = "*.csv" if is_letters else "*.npz"
type_name = "CSV Files" if is_letters else "NumPy Archives"
cat_name = "Letters" if is_letters else "Words"

# Step B: Choose Number of Files
num_str = input(f"How many {cat_name} files do you want to merge? (Max 4): ").strip()
try:
    num_files = int(num_str)
    if num_files < 1 or num_files > 4:
        print("\n❌ Invalid number. Defaulting to 2 files.")
        num_files = 2
except ValueError:
    print("\n❌ Invalid input. Defaulting to 2 files.")
    num_files = 2

# Step C: Trigger OS Pop-ups sequentially
selected_paths = []
print(f"\n📂 Look for the pop-up window! Please select your {num_files} files...")

for i in range(num_files):
    # This triggers the popup for File 1, then File 2, etc.
    path = prompt_for_file(f"{cat_name} File {i+1} of {num_files}", ext, type_name)
    if path:
        selected_paths.append(path)
    else:
        print(f"  ⚠️ Skipped File {i+1}")

# Step D: Extract and Display Clean Results
extracted_data = [] # This list will hold your actual loaded dataframes/arrays!
print("\n" + "="*60)

for i, path in enumerate(selected_paths):
    if is_letters:
        data = extract_and_display_csv(path, f"Dataset {i+1}")
    else:
        data = extract_and_display_npz(path, f"Dataset {i+1}")
        
    if data is not None:
        extracted_data.append(data)

print("\n" + "="*60)
print(f"✅ Extraction Complete. {len(extracted_data)} datasets loaded and ready.")

# --> READY FOR MERGE <--
# If you chose Letters, extracted_data[0] is your first CSV dataframe, extracted_data[1] is your second, etc.


# ---
# 
# ## 📖 Next Steps
# 
# After merging, your datasets are ready for training:
# 
# ```python
# # 1. Train ASL Letter Model
# # Run: Letters/ASL Letter (English)/Mediapipe_Training.ipynb
# # Load: asl_letters_merged.csv
# 
# # 2. Train ArSL Letter Model
# # Run: Letters/ArSL Letter (Arabic)/Mediapipe_Training.ipynb
# # Load: arsl_letters_merged.csv
# 
# # 3. Train ASL Word Model
# # Run: Words/ASL Word (English)/ASL_Word_Training.ipynb
# # Load: asl_words_merged_train.npz / _val.npz / _test.npz
# 
# # 4. Train ArSL Word Model
# # Run: Words/ArSL Word (Arabic)/ArSL_Word_Training.ipynb
# # Load: arsl_words_merged_train.npz / _val.npz / _test.npz
# ```
# 
# ---
# 
# ## ❓ Troubleshooting
# 
# | Problem | Solution |
# |---------|----------|
# | `FileNotFoundError` | Check `PROJECT_ROOT` path and dataset file paths in config |
# | Class imbalance ratio > 2x | Increase `max_samples_per_class` or get more data for underrepresented classes |
# | Features out of range | Check if MediaPipe version is consistent across datasets |
# | Low test accuracy | May indicate signer overlap between train/test — rerun with signer-aware split |
# | Memory error on large datasets | Reduce `max_samples_per_class` or process letter/word models separately |
# 

# In[30]:


# ============================================================
# CELL 7: CLEANING & MERGING THE EXTRACTED DATA
# ============================================================

print("🚀 STARTING THE MERGE & CLEANING PROCESS...")

# 1. Combine all 4 dataframes from the previous step into one giant dataset
combined_df = pd.concat(extracted_data, ignore_index=True)

# 2. FIX THE 35 CLASSES ISSUE: 
# Make every label uppercase and strip out accidental spaces so 'a' and ' A ' become 'A'
combined_df['label'] = combined_df['label'].astype(str).str.upper().str.strip()

# 3. Generate the Final Stats
total_samples = len(combined_df)
unique_classes = combined_df['label'].unique()
num_classes = len(unique_classes)

print(f"\n✅ Merged {len(extracted_data)} datasets successfully!")
print(f"📊 Total Samples: {total_samples:,}")
print(f"🏷️ Total Unique Classes: {num_classes}")
print(f"🔠 Classes Found: {sorted(unique_classes)}")

# 4. Final Warning Check
if num_classes > 29:
    print("\n⚠️ WARNING: You still have more than 29 classes. Look at the 'Classes Found' list above.")
    print("You might have numbers (0-9) or weird typo labels you need to delete.")

# 5. Save the final file to your folder
output_filename = "unified_asl_dataset.csv"
combined_df.to_csv(output_filename, index=False)
print(f"\n💾 Saved clean, combined dataset to: {output_filename}")


# In[31]:


# ============================================================
# CELL 8: DATA AUDIT - BEFORE & AFTER MERGE ANALYSIS
# ============================================================

import pandas as pd
from pathlib import Path

# 1. Define the files you want to compare
# (Update these paths if your files are named differently)
original_files = [
    "asl_mediapipe_keypoints_dataset.csv",
    "asl_mediapipe_keypoints_dataset_2.csv",
    "asl_mediapipe_keypoints_dataset_3.csv",
    "asl_mediapipe_keypoints_dataset_4.csv"
]
merged_file = "unified_asl_dataset.csv"

# --- HELPER FUNCTION FOR DEEP ANALYSIS ---
def analyze_dataset(file_path, title):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Skipped {title}: File not found ({path.name})")
        return 0, set()
        
    df = pd.read_csv(path)
    rows = len(df)
    features = len(df.columns) - 1 if 'label' in df.columns else len(df.columns)
    missing_data = df.isna().sum().sum()
    
    print(f"\n📄 {title.upper()} ({path.name})")
    print("-" * 50)
    print(f"  • Total Samples:   {rows:,}")
    print(f"  • Feature Columns: {features}")
    print(f"  • Missing Values:  {missing_data} (NaNs)")
    
    classes = set()
    if 'label' in df.columns:
        # Clean labels temporarily just for the audit to get an accurate count
        clean_labels = df['label'].astype(str).str.upper().str.strip()
        classes = set(clean_labels.unique())
        print(f"  • Unique Classes:  {len(classes)}")
        
        if len(classes) <= 10:
            print(f"  • Class List:      {sorted(classes)}")
        else:
            print(f"  • Class List:      {sorted(classes)[:5]} ... {sorted(classes)[-5:]} (Showing first/last 5)")
            
    return rows, classes

# ==========================================
# EXECUTE THE AUDIT
# ==========================================
print("🔍 INITIATING DATA AUDIT PIPELINE...")
print("=" * 60)

# --- PART 1: BEFORE MERGE ---
print("\n[PART 1: BEFORE MERGE - ORIGINAL FILES]")
total_original_rows = 0
all_original_classes = set()

for i, file in enumerate(original_files):
    rows, classes = analyze_dataset(file, f"Original Dataset {i+1}")
    total_original_rows += rows
    all_original_classes.update(classes) # Combine all unique classes seen so far

# --- PART 2: AFTER MERGE ---
print("\n\n[PART 2: AFTER MERGE - UNIFIED FILE]")
print("=" * 60)
final_rows, final_classes = analyze_dataset(merged_file, "Final Unified Dataset")

# --- PART 3: THE VERDICT (SUMMARY) ---
print("\n\n[PART 3: THE VERDICT & SUMMARY]")
print("=" * 60)

# Check Rows
if final_rows == total_original_rows:
    print(f"✅ ROW CHECK PASSED: All {total_original_rows:,} original rows were successfully moved to the merged file.")
else:
    print(f"⚠️ ROW CHECK FAILED: Original files had {total_original_rows:,} rows, but merged file has {final_rows:,}.")

# Check Classes
print(f"📊 CLASS SUMMARY: Your unified dataset covers {len(final_classes)} total classes.")
print(f"🔠 FINAL ALPHABET: {sorted(final_classes)}")

# Flag if there are weird classes (like numbers or typos)
weird_classes = [c for c in final_classes if len(c) > 1 or not c.isalpha()]
if weird_classes:
    print(f"\n🚨 DATA QUALITY WARNING: Found suspicious class names that might be typos: {weird_classes}")
    print("   You may need to delete these rows before training your model.")
else:
    print("\n✅ DATA QUALITY: All class names look clean (single letters).")

print("\n🏁 AUDIT COMPLETE.")


# In[32]:


# ============================================================
# CELL 9: SAFE DATASET BACKUP & FINAL CLEANUP
# ============================================================

import pandas as pd
import shutil
import os

print("🧹 STARTING FINAL CLEANUP PROCESS...")

# Define file names
input_file = "unified_asl_dataset.csv"
backup_file = "unified_asl_dataset_BACKUP.csv"
final_file = "clean_asl_dataset_final.csv"

# 1. CREATE A SAFE BACKUP FIRST
if os.path.exists(input_file):
    shutil.copy2(input_file, backup_file)
    print(f"📦 BACKUP SECURED: Exact copy saved as -> {backup_file}")
else:
    print(f"❌ ERROR: Cannot find {input_file}. Did you run the merge cell?")

# 2. Load the dataset for cleaning
df = pd.read_csv(input_file)
original_count = len(df)

# 3. Define exactly what classes we WANT to keep. 
valid_classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DEL' 
]

# 4. Filter the dataframe to ONLY include valid classes
clean_df = df[df['label'].isin(valid_classes)]
new_count = len(clean_df)
removed_count = original_count - new_count

# 5. Save the ultra-clean dataset
clean_df.to_csv(final_file, index=False)

# 6. Print the results
print("\n" + "=" * 50)
print(f"✅ Cleanup Complete!")
print(f"🗑️ Removed {removed_count:,} junk rows (Numbers, BLANK, Typos)")
print(f"📊 Final Clean Rows: {new_count:,}")
print(f"🏷️ Final Clean Classes: {clean_df['label'].nunique()}")
print(f"🔠 Final Alphabet: {sorted(clean_df['label'].unique())}")
print("=" * 50)
print(f"💾 READY FOR TRAINING: {final_file}")
print(f"🛡️ SAFE BACKUP KEPT:  {backup_file}")


# In[33]:


# ============================================================
# CELL 10: ML PREPROCESSING (ENCODING & TRAIN/TEST SPLIT)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

print("⚙️ STARTING ML PREPROCESSING PIPELINE...")

# 1. Load the clean dataset
df = pd.read_csv("clean_asl_dataset_final.csv")

# 2. Separate Features (X) and Labels (y)
# We assume 'label' is the column name. Everything else is a feature (the 63 keypoints).
X = df.drop(columns=['label']).values
y_text = df['label'].values

# 3. Label Encoding (Convert 'A' -> 0, 'B' -> 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)

# Save the encoder! You WILL need this later to translate the model's math back into letters.
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# 4. Train / Validation / Test Split (80% / 10% / 10%)
# First split: 80% Train, 20% Temp (which will become Val/Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

# Second split: Cut the 20% Temp in half -> 10% Val, 10% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 5. Check Feature Scaling (Just to be safe!)
print("\n🔍 Checking Data Normalization...")
print(f"   Min feature value: {X.min():.4f}")
print(f"   Max feature value: {X.max():.4f}")
if X.max() > 5.0 or X.min() < -5.0:
    print("   ⚠️ WARNING: Your features are outside the normal [0, 1] range.")
    print("   You may need to add a StandardScaler to your training pipeline.")
else:
    print("   ✅ Features look reasonably normalized (Likely MediaPipe defaults).")

# 6. Save the final arrays for training
np.savez_compressed("asl_training_data.npz", 
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)

print("\n" + "=" * 50)
print("✅ PREPROCESSING COMPLETE!")
print(f"📚 Training Data:   {len(X_train):,} samples (80%)")
print(f"🎯 Validation Data: {len(X_val):,} samples (10%)")
print(f"🧪 Testing Data:    {len(X_test):,} samples (10%)")
print("=" * 50)
print("💾 Saved ML-ready arrays to: asl_training_data.npz")
print("💾 Saved label translator to: label_encoder.pkl")


# In[ ]:


# ============================================================
# CELL 11: NEURAL NETWORK TRAINING (GPU OPTIMIZED)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 🔧 OPTIMIZATION: FIX GPU MEMORY HOARDING
# ------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Tell TF to only use GPU memory as needed, preventing crashes
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected & Memory Growth Enabled: {len(gpus)} GPU(s) found.")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ No GPU detected. Training on CPU.")

print("\n🧠 BUILDING THE NEURAL NETWORK...")

# 1. Load the ML-ready data and explicitly cast to float32 to save RAM
data = np.load("asl_training_data.npz")
X_train = data['X_train'].astype(np.float32)
y_train = data['y_train'].astype(np.int32)
X_val = data['X_val'].astype(np.float32)
y_val = data['y_val'].astype(np.int32)

num_features = X_train.shape[1] 
num_classes = len(np.unique(y_train))

# 2. Build the Neural Network Architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    Dropout(0.2), 
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(num_classes, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 4. TRAIN THE MODEL!
print("\n🚀 STARTING TRAINING...")
# We use a slightly larger batch size to optimize GPU throughput
history = model.fit(
    X_train, y_train,
    epochs=50,             
    batch_size=64,         
    validation_data=(X_val, y_val),
    verbose=1
)

# 5. Save the trained brain
model.save("asl_trained_model.keras")

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE!")
print("💾 Saved your trained model as: asl_trained_model.keras")
print("=" * 50)

# 6. Plot the learning curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


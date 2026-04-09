"""
model_loader.py — Utilities for loading Keras models and label encoders.

Handles:
  - .h5 model loading with custom objects (TemporalAttention)
  - Label encoder extraction from CSV datasets
  - Word class and shared vocabulary loading
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .temporal_attention import TemporalAttention

logger = logging.getLogger(__name__)

# Force float32 policy to avoid mixed-precision issues on CPU
try:
    tf.keras.mixed_precision.set_global_policy("float32")
except Exception:
    pass


def load_keras_model(
    model_path: str | Path,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> tf.keras.Model:
    """Load a Keras .h5 model, registering TemporalAttention by default.

    Args:
        model_path: Path to the .h5 model file.
        custom_objects: Extra custom objects dict (merged with defaults).

    Returns:
        Loaded ``tf.keras.Model``.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    objects: Dict[str, Any] = {"TemporalAttention": TemporalAttention}
    if custom_objects:
        objects.update(custom_objects)

    logger.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(str(model_path), custom_objects=objects)
    logger.info("Model loaded — input: %s  output: %s", model.input_shape, model.output_shape)
    return model


def load_label_encoder_from_csv(
    csv_path: str | Path,
    label_column: str = "label",
) -> List[str]:
    """Extract sorted unique labels from a CSV file.

    Args:
        csv_path: Path to the CSV containing training data.
        label_column: Name of the column holding class labels.

    Returns:
        Sorted list of label strings.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=0)  # read header only first
    columns = list(df.columns)

    # Auto-detect label column
    if label_column in columns:
        col = label_column
    elif "label" in columns:
        col = "label"
    elif "class" in columns:
        col = "class"
    else:
        # Fallback: assume last column is the label
        col = columns[-1]
        logger.warning("Label column '%s' not found; using last column '%s'", label_column, col)

    df = pd.read_csv(csv_path, usecols=[col])
    labels = sorted(df[col].dropna().unique().astype(str).tolist())
    logger.info("Loaded %d labels from %s (column='%s')", len(labels), csv_path.name, col)
    return labels


def load_word_classes(csv_path: str | Path) -> Dict[int, int]:
    """Load word class mapping:  model_class_index → word_id.

    Args:
        csv_path: Path to ``asl_word_classes.csv``.

    Returns:
        dict mapping ``model_class_index`` (int) → ``word_id`` (int).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Word classes CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Try expected columns
    idx_col = "model_class_index" if "model_class_index" in df.columns else df.columns[0]
    wid_col = "word_id" if "word_id" in df.columns else df.columns[1]

    mapping = dict(zip(df[idx_col].astype(int), df[wid_col].astype(int)))
    logger.info("Loaded %d word classes from %s", len(mapping), csv_path.name)
    return mapping


def load_shared_vocabulary(csv_path: str | Path) -> Dict[int, Dict[str, str]]:
    """Load shared word vocabulary:  word_id → {english, arabic, category}.

    Args:
        csv_path: Path to ``shared_word_vocabulary.csv``.

    Returns:
        dict mapping ``word_id`` (int) → ``{english, arabic, category}``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Vocabulary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Auto-detect columns
    wid_col = "word_id" if "word_id" in df.columns else df.columns[0]
    en_col = "english" if "english" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    ar_col = "arabic" if "arabic" in df.columns else (df.columns[2] if len(df.columns) > 2 else None)
    cat_col = "category" if "category" in df.columns else (df.columns[-1] if len(df.columns) > 3 else None)

    vocab: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        wid = int(row[wid_col])
        vocab[wid] = {
            "english": str(row[en_col]) if en_col else "",
            "arabic": str(row[ar_col]) if ar_col else "",
            "category": str(row[cat_col]) if cat_col else "",
        }

    logger.info("Loaded vocabulary with %d entries from %s", len(vocab), csv_path.name)
    return vocab

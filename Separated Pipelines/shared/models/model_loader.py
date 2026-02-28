"""
model_loader.py — Load .h5 models and label encoders for both pipelines.

References model files from the *original* project directory (SLR Main/)
so no files need to be copied. Models are lazily loaded — each pipeline
only loads what it needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from shared.models.temporal_attention import TemporalAttention


def _ensure_float32_policy():
    """Force float32 compute policy so CPU inference never crashes."""
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass


def load_keras_model(model_path: Path, custom_objects: Optional[dict] = None) -> Any:
    """Load a .h5 Keras model, handling custom objects and mixed precision.

    Parameters
    ----------
    model_path : Path
        Absolute path to the .h5 file.
    custom_objects : dict, optional
        Extra custom layers / objects needed for deserialization.

    Returns
    -------
    tf.keras.Model
    """
    _ensure_float32_policy()

    objs = {"TemporalAttention": TemporalAttention}
    if custom_objects:
        objs.update(custom_objects)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path), custom_objects=objs)
    return model


def load_label_encoder_from_csv(
    csv_path: Path,
    label_column: str = "label",
) -> List[str]:
    """Extract unique sorted labels from a CSV column.

    Returns a list where index == model class index.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        # Try common alternatives
        for alt in ["class", "sign", "target", "gesture"]:
            if alt in df.columns:
                label_column = alt
                break
        else:
            raise KeyError(
                f"Column '{label_column}' not found. Available: {list(df.columns)}"
            )

    labels = sorted(df[label_column].dropna().unique().tolist())
    return labels


def load_word_classes(csv_path: Path) -> Dict[int, int]:
    """Load word class mapping: model_class_index → word_id.

    Expects CSV with columns: class_index, word_id (or similar).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Word classes CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Auto-detect column names
    idx_col = None
    wid_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if "index" in cl or "class_index" in cl:
            idx_col = c
        if "word_id" in cl or "wid" in cl:
            wid_col = c

    if idx_col and wid_col:
        return dict(zip(df[idx_col].astype(int), df[wid_col].astype(int)))

    # Fallback: assume first column is index, second is word_id
    return dict(zip(df.iloc[:, 0].astype(int), df.iloc[:, 1].astype(int)))


def load_shared_vocabulary(csv_path: Path) -> Dict[int, Dict[str, str]]:
    """Load shared bilingual vocabulary.

    Returns: {word_id: {"english": ..., "arabic": ..., "category": ...}}
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Vocabulary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    vocab: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        wid = int(row["word_id"])
        vocab[wid] = {
            "english": str(row.get("english", "")),
            "arabic": str(row.get("arabic", "")),
            "category": str(row.get("category", "")),
        }
    return vocab

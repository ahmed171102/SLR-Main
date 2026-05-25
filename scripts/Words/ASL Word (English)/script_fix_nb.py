import json

def fix_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # 1. Replace model loading
        if "def build_model():" in source:
            new_source = """import tensorflow as tf
import pandas as pd
import numpy as np
import os

MODEL_WEIGHTS = r'asl_word_lstm_model_final.h5' 
CLASSES_CSV   = r'asl_word_classes.csv'
SHARED_CSV    = r'shared_word_vocabulary.csv'
SCALER_STATS  = r'asl_scaler_stats.npz'

# Load Metadata
class_df = pd.read_csv(CLASSES_CSV)
num_classes = len(class_df)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63 

# Load Model
print(f"📂 Loading model from {MODEL_WEIGHTS}...")
try:
    model = tf.keras.models.load_model(MODEL_WEIGHTS)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load Scaler
print(f"📂 Loading scaler from {SCALER_STATS}...")
z = np.load(SCALER_STATS)
scaler_mean = z["mean"].astype(np.float32)
scaler_scale = z["scale"].astype(np.float32)
print("✅ Scaler loaded successfully!")

# Load Vocabulary Mapping
vocab_df = pd.read_csv(SHARED_CSV).dropna(subset=['wlasl_class'])
id_to_english = dict(zip(vocab_df['word_id'].astype(int), vocab_df['english']))
if 'model_class_index' in class_df.columns:
    index_to_word = {int(row['model_class_index']): id_to_english.get(int(row.get('word_id', row['source_class_id'])), row.get('label_name', "Unknown")) 
                     for _, row in class_df.iterrows()}
else:
    # Fallback
    index_to_word = {i: row.get('label_name', 'Unknown') for i, row in class_df.iterrows()}

print(f"🏷️ Loaded {len(index_to_word)} words.")
"""
            # Split back into lines
            cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]
            
        # 2. Fix the prediction loop missing variables & scaling
        if "proba = model(seq, training=False).numpy()[0]" in source:
            # We will use string replace to inject the scaler and variables
            old_seq_part = "seq = np.expand_dims(seq, axis=0)"
            new_seq_part = """# Scale features
            seq_flat = seq.reshape(-1, NUM_FEATURES)
            seq_scaled = (seq_flat - scaler_mean) / scaler_scale
            seq = seq_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)"""
            
            new_source_mod = source.replace(
                "seq = np.expand_dims(seq, axis=0)  # Shape will be (1, 30, 126)",
                new_seq_part
            )
            # if the comment was slightly different we also try without comment
            new_source_mod = new_source_mod.replace(
                "seq = np.expand_dims(seq, axis=0)",
                new_seq_part
            )
            
            old_pred_part = "pred_idx = np.argmax(proba)"
            new_pred_part = """pred_idx = np.argmax(proba)
                pred_conf = proba[pred_idx]
                pred_word = index_to_word.get(pred_idx, '?')"""
                
            new_source_mod = new_source_mod.replace(old_pred_part, new_pred_part)
            
            cell['source'] = [line + '\n' for line in new_source_mod.split('\n')][:-1]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    fix_notebook("ASL_Word_Live_Test (1).ipynb")

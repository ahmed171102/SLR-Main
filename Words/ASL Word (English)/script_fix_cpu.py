import os

def fix_live_cpu(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Inject scaler loading after model loading
    if "SCALER_STATS = OUTPUT_DIR / \"asl_scaler_stats.npz\"" not in content:
        content = content.replace(
            "CLASSES_CSV = OUTPUT_DIR / \"asl_word_classes.csv\"",
            "CLASSES_CSV = OUTPUT_DIR / \"asl_word_classes.csv\"\nSCALER_STATS = OUTPUT_DIR / \"asl_scaler_stats.npz\""
        )
        
    if "scaler_mean = " not in content:
        scaler_load = """print("[INFO] Loading scaler...")
try:
    z = np.load(str(SCALER_STATS))
    scaler_mean = z["mean"].astype(np.float32)
    scaler_scale = z["scale"].astype(np.float32)
    print("[OK] Scaler loaded")
except Exception as e:
    print(f"[ERROR] Scaler loading failed: {e}")
    scaler_mean = 0.0
    scaler_scale = 1.0"""
        
        content = content.replace(
            "print(\"[OK] Model loaded\")",
            f"print(\"[OK] Model loaded\")\n\n{scaler_load}"
        )
        
    # Fix scaling logic
    if "seq_scaled = (seq_flat - scaler_mean) / scaler_scale" not in content:
        old_pred = "res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()"
        new_pred = """seq_arr = np.array(sequence, dtype=np.float32)
        seq_flat = seq_arr.reshape(-1, NUM_FEATURES)
        seq_scaled = (seq_flat - scaler_mean) / scaler_scale
        seq_arr = seq_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        res = model(seq_arr, training=False)[0].numpy()"""
        
        content = content.replace(old_pred, new_pred)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    fix_live_cpu("live_test_cpu.py")

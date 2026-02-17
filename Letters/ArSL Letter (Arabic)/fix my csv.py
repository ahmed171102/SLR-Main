import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Use the file you believe is most complete. 
# We will fix it regardless of its state.
INPUT_FILENAME = r'M:\Term 9\Grad\Main\Sign-Language-Recognition-System-main\Sign-Language-Recognition-System-main\Sign_to_Sentence Project Main\ArSL Letter (Arabic)\arabic_final_training_data copy.csv' 
OUTPUT_FILENAME = 'FINAL_CLEAN_DATASET.csv'

print(f"🚀 Starting Deep Clean on {INPUT_FILENAME}...")

if not os.path.exists(INPUT_FILENAME):
    print(f"❌ Error: {INPUT_FILENAME} not found in this folder.")
    print("   Please make sure the file is in the same folder as this script.")
else:
    # 1. Load Data
    df = pd.read_csv(INPUT_FILENAME)
    print(f"   Original Shape: {df.shape}")

    # 2. Fix Columns (Remove '0', '1', '2'...)
    # We only want 'label' and 'x0'...'z20'
    bad_cols = [c for c in df.columns if c.isdigit()]
    if bad_cols:
        print(f"   🧹 Removing {len(bad_cols)} bad numbered columns...")
        df.drop(columns=bad_cols, inplace=True)
    
    # 3. Check for 'nothing' class
    if 'nothing' not in df['label'].unique():
        print("   🔧 Class 'nothing' is MISSING. Adding 300 empty samples...")
        
        # Create 300 rows of zeros
        new_data = []
        feature_cols = [c for c in df.columns if c != 'label']
        for _ in range(300):
            row = {'label': 'nothing'}
            for col in feature_cols:
                row[col] = 0.0
            new_data.append(row)
            
        df_nothing = pd.DataFrame(new_data)
        df = pd.concat([df, df_nothing], ignore_index=True)
    else:
        print("   ✅ Class 'nothing' is present.")

    # 4. Final Validation
    print(f"   Final Shape: {df.shape}")
    print(f"   Total Classes: {df['label'].nunique()}")
    
    # 5. Save
    df.to_csv(OUTPUT_FILENAME, index=False)
    print("\n" + "="*50)
    print(f"🎉 SUCCESS! Saved perfect file: {OUTPUT_FILENAME}")
    print("="*50)
    print(f"👉 Now update your notebook: CSV_PATH = '{OUTPUT_FILENAME}'")
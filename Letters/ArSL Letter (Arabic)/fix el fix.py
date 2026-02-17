import pandas as pd
import numpy as np
import os

# ==========================================
# 1. PASTE YOUR FILE PATH HERE
# ==========================================
# This should be the path to the 'arabic_fixed_final.csv' file you just created.
# If it's in the same folder, just use the filename.
INPUT_FILE_PATH = r'M:\Term 9\Grad\Main\Sign-Language-Recognition-System-main\Sign-Language-Recognition-System-main\Sign_to_Sentence Project Main\ArSL Letter (Arabic)\arabic_fixed_final.csv' 

# ==========================================
# 2. THE SCRIPT
# ==========================================
print(f"🚀 Processing file: {INPUT_FILE_PATH}...")

if not os.path.exists(INPUT_FILE_PATH):
    print("❌ ERROR: File not found. Please check the path.")
else:
    # Load the dataset
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"   Original Size: {len(df)} rows")
    
    # Check if 'nothing' class exists
    if 'nothing' in df['label'].unique():
        print("✅ 'nothing' class already exists. No changes needed.")
    else:
        print("⚠️ 'nothing' class is missing. Adding it now...")
        
        # --- WHAT IS BEING ADDED ---
        # We are adding 300 rows where:
        # label = 'nothing'
        # x0, y0, z0 ... x20, y20, z20 = 0.0 (All Zeros)
        
        num_new_samples = 300
        feature_columns = [col for col in df.columns if col != 'label']
        
        # Create a list of 300 rows filled with zeros
        new_data = []
        for _ in range(num_new_samples):
            # [label] + [0, 0, 0, ...]
            row_data = {'label': 'nothing'}
            for col in feature_columns:
                row_data[col] = 0.0
            new_data.append(row_data)
            
        # Convert to DataFrame
        df_nothing = pd.DataFrame(new_data)
        
        # Merge with original data
        df_final = pd.concat([df, df_nothing], ignore_index=True)
        
        # Shuffle the data
        df_final = df_final.sample(frac=1).reset_index(drop=True)
        
        # Save to new file
        OUTPUT_FILE = 'arabic_fixed_with_nothing.csv'
        df_final.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*50)
        print("🎉 SUCCESS! Added 300 'nothing' samples.")
        print(f"   Saved new file: '{OUTPUT_FILE}'")
        print(f"   New Class Count: {df_final['label'].nunique()} (Should be 34)")
        print("="*50)
        print(f"👉 Now set CSV_PATH = '{OUTPUT_FILE}' in your training notebook.")
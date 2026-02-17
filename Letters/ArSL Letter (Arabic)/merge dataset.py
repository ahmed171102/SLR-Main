import pandas as pd
import numpy as np
import os

# ⚠️ PASTE YOUR FULL PATHS HERE (Keep the 'r' before the quotes!)
# Example: r"C:\Users\Adel\Downloads\Arabic_Dataset.csv"
ARABIC_FILE = r"M:\Term 9\Grad\Main\Sign-Language-Recognition-System-main\Sign-Language-Recognition-System-main\Sign_to_Sentence Project Main\ArSL Letter (Arabic)\Arabic Sign Language Letters Dataset.csv"
ENGLISH_FILE = r"Sign-Language-Recognition-System-main/Sign-Language-Recognition-System-main/Sign_to_Sentence Project Main/ArSL Letter (Arabic)/asl_mediapipe_keypoints_dataset.csv"

# Output name
OUTPUT_FILE = "arabic_ready_for_training.csv"

print("🚀 Starting Merger...")

# --- 1. Load Data ---
if os.path.exists(ARABIC_FILE) and os.path.exists(ENGLISH_FILE):
    df_ar = pd.read_csv(ARABIC_FILE)
    df_en = pd.read_csv(ENGLISH_FILE)
    print(f"✅ Loaded Arabic: {len(df_ar)} rows")
    print(f"✅ Loaded English: {len(df_en)} rows")
    
    # --- 2. Fix Column Names ---
    # Arabic usually has 'letter', English has 'label' or 'class'
    if 'letter' in df_ar.columns: 
        df_ar.rename(columns={'letter': 'label'}, inplace=True)
    
    # Check English columns (it might be 'class' or 'label')
    if 'class' in df_en.columns: 
        df_en.rename(columns={'class': 'label'}, inplace=True)
    
    # --- 3. Extract Special Classes ---
    # We want these specific gestures from the English file
    special_classes = ['space', 'del', 'nothing']
    
    # Filter for ONLY these classes
    df_specials = df_en[df_en['label'].isin(special_classes)].copy()
    
    if df_specials.empty:
        print("⚠️ WARNING: No 'space', 'del', or 'nothing' found in English file.")
        print("   Columns found:", df_en.columns)
        print("   Labels found:", df_en['label'].unique()[:10])
    else:
        # Balance them (Take 300 of each so they don't dominate)
        df_specials = df_specials.groupby('label').head(300)
        print(f"✅ Extracted {len(df_specials)} special samples.")

        # --- 4. Merge & Save ---
        df_final = pd.concat([df_ar, df_specials], ignore_index=True)
        df_final = df_final.sample(frac=1).reset_index(drop=True) # Shuffle
        
        # Save to the SAME folder as the Arabic file
        save_path = os.path.join(os.path.dirname(ARABIC_FILE), OUTPUT_FILE)
        df_final.to_csv(save_path, index=False)
        
        print("\n" + "="*50)
        print(f"🎉 SUCCESS! Created: {save_path}")
        print(f"📊 Total Classes: {df_final['label'].nunique()} (Target: 34)")
        print("="*50)

else:
    print("❌ Error: Still can't find files.")
    print(f"   Checked: {ARABIC_FILE}")
    print(f"   Checked: {ENGLISH_FILE}")
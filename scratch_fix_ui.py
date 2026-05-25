import json

file_path = r'm:\Term 10\Grad\SLR Main\Unified_Dataset_Merger.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that contains 'CELL 5 & 6: DYNAMIC DATASET EXTRACTION'
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell['source']:
        text = "".join(cell['source'])
        if 'DYNAMIC DATASET EXTRACTION' in text and 'tkinter' in text:
            target_idx = i
            break

if target_idx != -1:
    new_source = """# ============================================================
# CELL 5 & 6: DYNAMIC DATASET EXTRACTION & VALIDATION (UI OPTIMIZED)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import ipywidgets as widgets
from IPython.display import clear_output, display

# ------------------------------------------------------------
# 1. VALIDATION FUNCTIONS (Clean Text Output)
# ------------------------------------------------------------
def extract_and_display_csv(file_path, dataset_name):
    if not file_path: return None
    path = Path(file_path.strip('\"\\'')) # Strip quotes if pasted
    
    if not path.exists():
        print(f"  ❌ Error: File not found -> {path}")
        return None
        
    print(f"\\n📊 --- {dataset_name} ---")
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
    path = Path(file_path.strip('\"\\''))
    
    if not path.exists():
        print(f"  ❌ Error: File not found -> {path}")
        return None
        
    print(f"\\n📈 --- {dataset_name} ---")
    data = np.load(path)
    X, y = data['X'], data['y']
    
    print(f"  • File:      {path.name}")
    print(f"  • Sequences: {len(X):,}")
    print(f"  • Shape:     {X.shape} (Sequences, Frames, Features)")
    print(f"  • Classes:   {len(np.unique(y))}")
    
    return X, y

# ------------------------------------------------------------
# 2. IPYWIDGETS INTERACTIVE DASHBOARD
# ------------------------------------------------------------
cat_dropdown = widgets.Dropdown(
    options=[('Letters (CSV Files)', 'letters'), ('Words (NPZ Files)', 'words')],
    value='letters',
    description='Category:',
    style={'description_width': 'initial'}
)

# Text inputs for paths instead of tkinter popups
path_inputs = [
    widgets.Text(placeholder='Paste absolute path to dataset here...', description=f'File {i+1}:', layout=widgets.Layout(width='90%'))
    for i in range(4)
]

btn_extract = widgets.Button(description='🚀 Extract & Validate', button_style='primary', icon='cogs', layout=widgets.Layout(width='auto'))
out_logs = widgets.Output()

# Global variable to pass to the next cell
extracted_data = []

def on_extract(b):
    with out_logs:
        clear_output()
        global extracted_data
        extracted_data = []
        
        is_letters = (cat_dropdown.value == 'letters')
        paths = [inp.value.strip() for inp in path_inputs if inp.value.strip() != '']
        
        if not paths:
            print("❌ ERROR: Please paste at least one file path!")
            return
            
        print("🚀 STARTING DYNAMIC EXTRACTION PIPELINE")
        print("="*60)
        
        for i, p in enumerate(paths):
            if is_letters:
                data = extract_and_display_csv(p, f"Dataset {i+1}")
            else:
                data = extract_and_display_npz(p, f"Dataset {i+1}")
                
            if data is not None:
                extracted_data.append(data)
                
        print("\\n" + "="*60)
        print(f"✅ Extraction Complete. {len(extracted_data)} datasets loaded and ready for the next cell.")

btn_extract.on_click(on_extract)

ui = widgets.VBox([
    widgets.HTML("<h3>📂 Dynamic Dataset Extraction Panel</h3>"),
    widgets.HTML("<p><i>Paste the full paths to your files below to avoid UI freezing issues. Leave unused boxes empty.</i></p>"),
    cat_dropdown,
    widgets.HTML("<br><b>Dataset Paths:</b>"),
    *path_inputs,
    widgets.HTML("<br>"),
    btn_extract,
    out_logs
], layout=widgets.Layout(padding='15px', border='2px solid #2196F3', border_radius='10px', background_color='#f0f8ff'))

display(ui)
"""
    nb['cells'][target_idx]['source'] = [line + '\\n' for line in new_source.split('\\n')]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook optimized successfully!")
else:
    print("Could not find the target cell.")

import json
import io

file_path = r'm:\Term 10\Grad\SLR Main\Unified_Dataset_Merger.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find CELL 5: EXECUTE MERGES and CELL 6: QUALITY VALIDATION
cell_5_idx = -1
cell_6_idx = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell['source']:
        if 'CELL 5: EXECUTE MERGES' in cell['source'][1] or 'CELL 5: EXECUTE MERGES' in cell['source'][0]:
            cell_5_idx = i
        if 'CELL 6: QUALITY VALIDATION' in cell['source'][1] or 'CELL 6: QUALITY VALIDATION' in cell['source'][0]:
            cell_6_idx = i

print(f"Cell 5 is at index {cell_5_idx}")
print(f"Cell 6 is at index {cell_6_idx}")

new_ui_cell_source = """# ============================================================
# CELL 5 & 6: INTERACTIVE DASHBOARD (EXECUTION & VALIDATION)
# ============================================================

import ipywidgets as widgets
from IPython.display import display, clear_output

def validate_letter_dataset(csv_path):
    \"\"\"
    Validate a letter dataset CSV.
    \"\"\"
    print(f"\\n📋 VALIDATING: {csv_path.name}")
    
    if not csv_path.exists():
        print(f"   ❌ File not found")
        return False
    
    df = pd.read_csv(csv_path)
    
    # Check columns
    num_feature_cols = len(df.columns) - 1  # Minus label column
    print(f"   📊 Samples: {len(df):,}")
    print(f"   📊 Features: {num_feature_cols}")
    print(f"   📊 Classes: {df['label'].nunique()}")
    
    # Check for NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"   ❌ Found {nan_count} NaN values")
        return False
    else:
        print(f"   ✅ No NaN values")
    
    # Check feature ranges
    feature_cols = [c for c in df.columns if c != 'label']
    min_val = df[feature_cols].min().min()
    max_val = df[feature_cols].max().max()
    print(f"   📊 Feature range: [{min_val:.4f}, {max_val:.4f}]")
    
    if min_val < -5 or max_val > 5:
        print(f"   ⚠️  WARNING: Features out of expected range [0, 1] — may need normalization")
    else:
        print(f"   ✅ Features in reasonable range")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    print(f"   📊 Class distribution:")
    print(f"      Min samples/class: {class_counts.min()}")
    print(f"      Max samples/class: {class_counts.max()}")
    print(f"      Imbalance ratio: {class_counts.max() / class_counts.min():.2f}x")
    
    if class_counts.max() / class_counts.min() > 1.5:
        print(f"   ⚠️  WARNING: Class imbalance > 1.5x — consider more balancing")
    else:
        print(f"   ✅ Classes well-balanced")
    
    print(f"   ✅ VALIDATION PASSED")
    return True

def validate_word_dataset(npz_path):
    \"\"\"
    Validate a word dataset NPZ.
    \"\"\"
    print(f"\\n📋 VALIDATING: {npz_path.name}")
    
    if not npz_path.exists():
        print(f"   ❌ File not found")
        return False
    
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    
    print(f"   📊 Sequences: {len(X):,}")
    print(f"   📊 Shape: {X.shape} (sequences, frames, features)")
    print(f"   📊 Classes: {len(np.unique(y))}")
    print(f"   📊 Class range: {y.min()}-{y.max()}")
    
    # Check for NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"   ❌ Found {nan_count} NaN values")
        return False
    else:
        print(f"   ✅ No NaN values")
    
    # Check feature ranges
    print(f"   📊 Feature range: [{X.min():.4f}, {X.max():.4f}]")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"   📊 Class distribution:")
    print(f"      Min samples/class: {counts.min()}")
    print(f"      Max samples/class: {counts.max()}")
    print(f"      Imbalance ratio: {counts.max() / counts.min():.2f}x")
    
    print(f"   ✅ VALIDATION PASSED")
    return True

# Create UI elements for merging
chk_asl_letters = widgets.Checkbox(value=False, description='🔤 ASL Letters (CSV)', indent=False)
chk_arsl_letters = widgets.Checkbox(value=False, description='🔤 ArSL Letters (CSV)', indent=False)
chk_asl_words = widgets.Checkbox(value=False, description='📖 ASL Words (NPZ)', indent=False)
chk_arsl_words = widgets.Checkbox(value=False, description='📖 ArSL Words (NPZ)', indent=False)

btn_run_merge = widgets.Button(description='🚀 Run Selected Merges', button_style='success', icon='play', layout=widgets.Layout(width='auto'))
btn_run_val = widgets.Button(description='🔍 Run Validations', button_style='info', icon='check', layout=widgets.Layout(width='auto'))
btn_clear = widgets.Button(description='🧹 Clear Output', button_style='warning', icon='eraser', layout=widgets.Layout(width='auto'))

out_logs = widgets.Output()

def run_merges(b):
    with out_logs:
        models_to_merge = {
            'asl_letters': chk_asl_letters.value,
            'arsl_letters': chk_arsl_letters.value,
            'asl_words': chk_asl_words.value,
            'arsl_words': chk_arsl_words.value,
        }
        
        if not any(models_to_merge.values()):
            print("⚠️ Please select at least one dataset to merge.")
            return

        print(f"\\n{'='*70}")
        print('🚀 STARTING UNIFIED DATASET MERGE')
        print(f"{'='*70}\\n")
        
        results = {}
        for model_key, should_merge in models_to_merge.items():
            if not should_merge:
                continue
            
            config = MERGE_CONFIG[model_key]
            model_type = config['model_type']
            
            try:
                if model_type == 'letters':
                    result = merge_letter_dataset(model_key)
                elif model_type == 'words':
                    result = merge_word_dataset(model_key)
                results[model_key] = result
            except Exception as e:
                print(f"❌ ERROR in {model_key}: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\\n{'='*70}")
        print('✅ MERGE PIPELINE COMPLETE')
        print(f"{'='*70}")
        print(f"\\n📊 Summary:")
        for model_key in models_to_merge:
            if models_to_merge[model_key]:
                status = "✅ Done" if model_key in results else "❌ Failed"
                print(f"   {model_key.upper()}: {status}")

def run_validations(b):
    with out_logs:
        print(f"\\n{'='*70}")
        print('🔍 VALIDATING MERGED DATASETS')
        print(f"{'='*70}\\n")
        
        for model_key, config in MERGE_CONFIG.items():
            output = config['output']
            
            if config['model_type'] == 'letters':
                if output.exists():
                    validate_letter_dataset(output)
                else:
                    print(f"\\n📋 SKIPPING: {output.name} (not found)")
            elif config['model_type'] == 'words':
                if output.exists():
                    validate_word_dataset(output)
                else:
                    print(f"\\n📋 SKIPPING: {output.name} (not found)")

        print(f"\\n{'='*70}")
        print('✅ ALL VALIDATIONS COMPLETE')
        print(f"{'='*70}")

def clear_logs(b):
    with out_logs:
        clear_output()

btn_run_merge.on_click(run_merges)
btn_run_val.on_click(run_validations)
btn_clear.on_click(clear_logs)

# Layout
controls_vbox = widgets.VBox([
    widgets.HTML("<h3>🧩 Dataset Merge Control Panel</h3>"),
    widgets.HTML("<b>Select Datasets to Merge:</b>"),
    widgets.HBox([chk_asl_letters, chk_arsl_letters, chk_asl_words, chk_arsl_words]),
    widgets.HTML("<br><b>Actions:</b>"),
    widgets.HBox([btn_run_merge, btn_run_val, btn_clear])
], layout=widgets.Layout(padding='15px', border='2px solid #4CAF50', border_radius='10px', background_color='#f9f9f9'))

display(controls_vbox, out_logs)
"""

if cell_5_idx != -1 and cell_6_idx != -1:
    # We will replace CELL 5 with the new UI cell
    nb['cells'][cell_5_idx]['source'] = [line + '\\n' for line in new_ui_cell_source.split('\\n')]
    
    # We will delete everything from the start of CELL 9 (Quality checks md) up to and including CELL 10 (Quality checks code)
    # Actually, let's just delete the original Cell 6 (Quality checks code), and its preceding markdown cell if there is one.
    # The cell before cell_6_idx is likely the markdown "## ✅ Quality Checks".
    del nb['cells'][cell_6_idx]
    
    if cell_6_idx - 1 > cell_5_idx and nb['cells'][cell_6_idx - 1]['cell_type'] == 'markdown':
        if 'Quality Checks' in ''.join(nb['cells'][cell_6_idx - 1]['source']):
            del nb['cells'][cell_6_idx - 1]
            
    # And maybe rename the markdown "Run the Merges"
    if cell_5_idx - 1 >= 0 and nb['cells'][cell_5_idx - 1]['cell_type'] == 'markdown':
        md_text = ''.join(nb['cells'][cell_5_idx - 1]['source'])
        if 'Run the Merges' in md_text:
            nb['cells'][cell_5_idx - 1]['source'] = ["---\\n", "\\n", "## ▶️ Interactive Dashboard\\n", "\\n", "Select your datasets, merge them, and validate the outputs using the control panel below."]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully updated notebook with UI cells!")
else:
    print("Could not find Cell 5 or Cell 6")

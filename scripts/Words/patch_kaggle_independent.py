"""
Script to patch the ArSL Kaggle notebook to train independently
without shared_word_vocabulary.csv.

It reads class labels directly from the KArSL-502 folder structure.
"""
import json, sys, io, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB_PATH = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle.ipynb'
OUT_PATH = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle_Independent.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_out = copy.deepcopy(nb)

# ===================================================================
# CELL 3 (Config) — Remove SHARED_CSV references, keep KARSL_ROOT
# ===================================================================
new_cell3 = r"""# ===============================
# CELL 3: CONFIGURATION / PATHS
# ===============================
# INDEPENDENT MODE: No shared_word_vocabulary.csv needed!
# Classes are auto-discovered from KArSL-502 folder structure.

IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    # ===== KAGGLE PATHS =====
    KAGGLE_INPUT = Path('/kaggle/input')
    KAGGLE_OUTPUT = Path('/kaggle/working')

    # Update this dataset name to match YOUR Kaggle dataset upload:
    KARSL_DATASET = KAGGLE_INPUT / 'karsl-502'

    KARSL_ROOT = KARSL_DATASET
    OUTPUT_DIR = KAGGLE_OUTPUT
else:
    # ===== LOCAL PATHS =====
    PROJECT_ROOT = Path(r'M:/Term 10/Grad')
    SLR_MAIN = PROJECT_ROOT / 'SLR Main'
    WORDS_ROOT = SLR_MAIN / 'Words'
    KARSL_ROOT = WORDS_ROOT / 'Datasets/KArSL_502'
    OUTPUT_DIR = WORDS_ROOT / 'ArSL Word (Arabic)'

# ===== TWO-HAND SEQUENCE PARAMETERS =====
SEQUENCE_LENGTH = 30        # frames per sample
NUM_HANDS = 2               # detect both hands
LANDMARKS_PER_HAND = 63     # 21 landmarks x 3 (x, y, z)
NUM_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND  # 126 features

# ===== FULL TRAINING HYPERPARAMETERS =====
BATCH_SIZE      = 64
EPOCHS          = 150
LEARNING_RATE   = 5e-4
LSTM_UNITS_1    = 256
LSTM_UNITS_2    = 128
LSTM_UNITS_3    = 64
DENSE_UNITS     = 256
DROPOUT_RATE    = 0.4
LABEL_SMOOTH    = 0.1
GRAD_CLIP_NORM  = 1.0
L2_REG          = 1e-4
TEST_SIZE       = 0.4       # val+test fraction -> 60/20/20

# If True, load pre-extracted .npy/.csv keypoints (faster)
# If False, extract from raw .mp4 videos using MediaPipe
USE_PREEXTRACTED_KEYPOINTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) if not IS_KAGGLE else None

# Verify paths
for name, path in [('KArSL root', KARSL_ROOT)]:
    status = 'FOUND' if path.exists() else 'NOT FOUND'
    print(f'{status} — {name}: {path}')

print(f'\nOutput dir      : {OUTPUT_DIR}')
print(f'\nSequence length : {SEQUENCE_LENGTH}')
print(f'Hands           : {NUM_HANDS}')
print(f'Features/frame  : {NUM_FEATURES}')
print(f'Batch size      : {BATCH_SIZE}')
print(f'Max epochs      : {EPOCHS}')
print(f'Learning rate   : {LEARNING_RATE}')
print(f'LSTM units      : {LSTM_UNITS_1}/{LSTM_UNITS_2}/{LSTM_UNITS_3}')
print(f'Dense units     : {DENSE_UNITS}')
print(f'Pre-extracted   : {USE_PREEXTRACTED_KEYPOINTS}')
print(f'Running on      : {"Kaggle" if IS_KAGGLE else "Local"}')
print(f'Mode            : INDEPENDENT (no shared vocab needed)')
"""

# ===================================================================
# CELL 4 (Vocab) — Auto-discover classes from KArSL folder structure
# ===================================================================
new_cell4 = r"""# ===============================
# CELL 4: AUTO-DISCOVER VOCABULARY FROM DATASET
# ===============================
# Instead of reading shared_word_vocabulary.csv, we scan the KArSL-502
# folder structure and use each subfolder name as a class.
# This makes the notebook fully independent.

print('=' * 60)
print('AUTO-DISCOVERING VOCABULARY FROM KArSL FOLDER STRUCTURE')
print('=' * 60)

if not KARSL_ROOT.exists():
    print(f'\nKArSL dataset NOT FOUND at: {KARSL_ROOT}')
    print('Please download KArSL-502 from Kaggle:')
    print('https://www.kaggle.com/datasets/yousefelkilany/karsl-502')
    raise FileNotFoundError(f'KArSL dataset not found: {KARSL_ROOT}')

# Discover all class folders (each folder = one sign class)
class_folders = sorted([
    d for d in KARSL_ROOT.iterdir()
    if d.is_dir() and not d.name.startswith('.')
])

print(f'\nFound {len(class_folders)} class folders in KArSL dataset')

# Build mappings
# folder name -> class label (use folder name directly as label)
target_karsl_classes = []
id_to_label = {}

for folder in class_folders:
    # Try to parse as integer class ID first
    try:
        class_id = int(folder.name)
    except ValueError:
        class_id = folder.name  # use folder name as-is if not numeric
    
    target_karsl_classes.append(class_id)
    id_to_label[class_id] = str(class_id)

# Simple mappings that downstream cells expect
# Since we don't have English/Arabic translations, use class IDs as labels
id_to_english = {cid: str(cid) for cid in target_karsl_classes}
id_to_arabic = {cid: str(cid) for cid in target_karsl_classes}
karsl_to_wordid = {cid: cid for cid in target_karsl_classes}

# Count videos per class for summary
total_videos = 0
for folder in class_folders:
    n_vids = len(list(folder.glob('*.mp4'))) + len(list(folder.glob('*.avi')))
    total_videos += n_vids

print(f'Total classes    : {len(target_karsl_classes)}')
print(f'Total videos     : {total_videos}')
print(f'Avg videos/class : {total_videos / max(len(target_karsl_classes), 1):.1f}')
print(f'\nSample classes   : {target_karsl_classes[:15]}...')
print(f'\nVocabulary mode  : INDEPENDENT (auto-discovered from folders)')
"""

# ===================================================================
# Apply patches to the notebook cells
# ===================================================================
# Find the actual cell indices for Cell 3 (config) and Cell 4 (vocab)
code_cell_count = 0
cell3_idx = None
cell4_idx = None

for i, cell in enumerate(nb_out['cells']):
    if cell.get('cell_type') == 'code':
        code_cell_count += 1
        src = "".join(cell.get("source", []))
        if 'CELL 3: CONFIGURATION' in src or 'CELL 3:' in src:
            cell3_idx = i
        elif 'CELL 4: LOAD SHARED VOCABULARY' in src or 'CELL 4:' in src:
            cell4_idx = i

print(f"Cell 3 (config) found at nb index: {cell3_idx}")
print(f"Cell 4 (vocab) found at nb index: {cell4_idx}")

# Patch Cell 3
if cell3_idx is not None:
    nb_out['cells'][cell3_idx]['source'] = [new_cell3]
    nb_out['cells'][cell3_idx]['outputs'] = []
    print("  Patched Cell 3 (config) -- removed SHARED_CSV")

# Patch Cell 4
if cell4_idx is not None:
    nb_out['cells'][cell4_idx]['source'] = [new_cell4]
    nb_out['cells'][cell4_idx]['outputs'] = []
    print("  Patched Cell 4 (vocab) -- auto-discover from folders")

# ===================================================================
# Patch Cell 10 (Evaluation) — remove per-category accuracy plot
# since we no longer have category info from shared vocab
# ===================================================================
for i, cell in enumerate(nb_out['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = "".join(cell.get("source", []))
    if 'CELL 10: EVALUATION' in src or 'EVALUATION & VISUALIZATION DASHBOARD' in src:
        # Replace the category accuracy section with a safe fallback
        old_cat = "cat_map = dict(zip(vocab_df['word_id'].astype(int), vocab_df['category']))"
        new_cat = "# Per-category accuracy skipped (independent mode — no category data)\ncat_map = {}"
        
        new_src = src.replace(old_cat, new_cat)
        
        # Also fix category section to handle empty cat_map
        old_cat_check = "cat_names = sorted(category_total.keys())"
        new_cat_check = "cat_names = sorted(category_total.keys())\nif not cat_names:\n    print('Per-category accuracy: skipped (no category data in independent mode)')"
        new_src = new_src.replace(old_cat_check, new_cat_check)
        
        nb_out['cells'][i]['source'] = [new_src]
        nb_out['cells'][i]['outputs'] = []
        print(f"  Patched Cell at index {i} (evaluation) -- category fallback added")
        break

# ===================================================================
# Save the new notebook
# ===================================================================
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb_out, f, ensure_ascii=False, indent=1)

print(f"\nSaved independent notebook to:\n  {OUT_PATH}")
print("\nThis notebook no longer requires shared_word_vocabulary.csv!")
print("It only needs the KArSL-502 dataset on Kaggle.")

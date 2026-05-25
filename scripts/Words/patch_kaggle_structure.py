import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle_Independent.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ===================================================================
# NEW Cell 4 (Vocab) — handles nested KArSL structure: 01/01/train/
# ===================================================================
new_cell4 = r"""# ===============================
# CELL 4: AUTO-DISCOVER VOCABULARY FROM DATASET
# ===============================
# Scans the KArSL-502 folder structure and auto-discovers all classes.
# Handles the nested structure: KARSL-502/{class_id}/{class_id}/train/
# No external vocabulary file needed.

print('=' * 60)
print('AUTO-DISCOVERING VOCABULARY FROM KArSL FOLDER STRUCTURE')
print('=' * 60)

if not KARSL_ROOT.exists():
    print(f'\nKArSL dataset NOT FOUND at: {KARSL_ROOT}')
    print('Please download KArSL-502 from Kaggle:')
    print('https://www.kaggle.com/datasets/yousefelkilany/karsl-502')
    raise FileNotFoundError(f'KArSL dataset not found: {KARSL_ROOT}')

# Discover all top-level class folders
class_folders = sorted([
    d for d in KARSL_ROOT.iterdir()
    if d.is_dir() and not d.name.startswith('.')
])

print(f'\nFound {len(class_folders)} top-level class folders')

# Detect the actual folder structure by checking the first folder
sample_folder = class_folders[0] if class_folders else None
NESTED_STRUCTURE = False
HAS_TRAIN_TEST_SPLIT = False

if sample_folder:
    # Check if there's a nested subfolder with the same name (e.g. 01/01/)
    inner = sample_folder / sample_folder.name
    if inner.is_dir():
        NESTED_STRUCTURE = True
        # Check if inner folder has train/test subfolders
        if (inner / 'train').is_dir() or (inner / 'test').is_dir():
            HAS_TRAIN_TEST_SPLIT = True

print(f'Nested structure : {NESTED_STRUCTURE}  (e.g. 01/01/)')
print(f'Train/Test split : {HAS_TRAIN_TEST_SPLIT}  (e.g. 01/01/train/)')

# Build class list and mappings
target_karsl_classes = []
id_to_label = {}

for folder in class_folders:
    try:
        class_id = int(folder.name)
    except ValueError:
        class_id = folder.name
    
    target_karsl_classes.append(class_id)
    id_to_label[class_id] = str(class_id)

# Simple mappings for downstream cells
id_to_english = {cid: str(cid) for cid in target_karsl_classes}
id_to_arabic = {cid: str(cid) for cid in target_karsl_classes}
karsl_to_wordid = {cid: cid for cid in target_karsl_classes}

# Count videos by scanning the actual structure
total_videos = 0
for folder in class_folders:
    # Use rglob to find all videos recursively (handles any nesting)
    n_vids = len(list(folder.rglob('*.mp4'))) + len(list(folder.rglob('*.avi')))
    total_videos += n_vids

print(f'\nTotal classes    : {len(target_karsl_classes)}')
print(f'Total videos     : {total_videos}')
print(f'Avg videos/class : {total_videos / max(len(target_karsl_classes), 1):.1f}')
print(f'\nSample classes   : {target_karsl_classes[:15]}...')
print(f'\nVocabulary mode  : INDEPENDENT (auto-discovered from folders)')
"""

# ===================================================================
# NEW Cell 6 (Build Dataset) — handles nested structure
# ===================================================================
new_cell6 = r"""# ============================================
# CELL 6: BUILD DATASET (or Load Cached)
# ============================================
# Handles the nested KArSL structure:
#    KARSL-502/{class_id}/{class_id}/train/*.mp4
#    KARSL-502/{class_id}/{class_id}/test/*.mp4

print('=' * 60)
print('BUILDING ARABIC WORD DATASET (TWO-HAND)')
print('=' * 60)

NPZ_PATH = OUTPUT_DIR / 'arsl_word_sequences_2hand.npz'

if NPZ_PATH.exists():
    print(f'\nCached data found: {NPZ_PATH}')
    data = np.load(NPZ_PATH)
    X, y = data['X'], data['y']
    print(f'   X shape : {X.shape}')
    print(f'   y shape : {y.shape}')
    print(f'   Classes : {len(np.unique(y))}')
    print('   Loaded from cache — skipping extraction')
else:
    if not KARSL_ROOT.exists():
        print(f'\nKArSL dataset NOT FOUND at: {KARSL_ROOT}')
        raise FileNotFoundError(f'KArSL dataset not found: {KARSL_ROOT}')

    print(f'\nLoading KArSL data from: {KARSL_ROOT}')
    print(f'Nested structure: {NESTED_STRUCTURE} | Train/Test split: {HAS_TRAIN_TEST_SPLIT}')
    start_time = time.time()

    X_list, y_list = [], []
    found_classes, empty_classes = 0, 0

    for karsl_class in tqdm(target_karsl_classes, desc='Loading KArSL classes'):
        word_id = int(karsl_to_wordid[karsl_class])

        # Build the correct path based on detected structure
        class_dir = KARSL_ROOT / str(karsl_class)
        if not class_dir.exists():
            # Try zero-padded names
            for fmt in [f'{karsl_class:02d}', f'{karsl_class:03d}', f'{karsl_class:04d}']:
                alt = KARSL_ROOT / fmt
                if alt.exists():
                    class_dir = alt
                    break

        if not class_dir.exists():
            empty_classes += 1
            continue

        found_classes += 1

        # Collect ALL data files recursively (handles 01/01/train/ structure)
        # Using rglob scans through all nested subfolders automatically
        if USE_PREEXTRACTED_KEYPOINTS:
            files = list(class_dir.rglob('*.npy')) + list(class_dir.rglob('*.csv'))
        else:
            files = list(class_dir.rglob('*.mp4')) + list(class_dir.rglob('*.avi'))

        if not files:
            # Fallback: try all types recursively
            files = (list(class_dir.rglob('*.npy')) + 
                     list(class_dir.rglob('*.csv')) + 
                     list(class_dir.rglob('*.mp4')) +
                     list(class_dir.rglob('*.avi')))

        for fp in files:
            seq = None
            try:
                if fp.suffix.lower() == '.npy':
                    arr = np.load(fp)
                    arr_2h = adapt_preextracted_to_2hand(arr)
                    seq = pad_or_sample(arr_2h)
                elif fp.suffix.lower() == '.csv':
                    arr = pd.read_csv(fp).values
                    arr_2h = adapt_preextracted_to_2hand(arr)
                    seq = pad_or_sample(arr_2h)
                elif fp.suffix.lower() in ['.mp4', '.avi']:
                    seq = extract_from_video_2hand(fp)
            except Exception:
                continue

            if seq is None:
                continue

            # Skip blank sequences (<20% hand detection)
            blank_ratio = np.sum(np.all(seq == 0, axis=1)) / len(seq)
            if blank_ratio > 0.8:
                continue

            X_list.append(seq)
            y_list.append(word_id)

    elapsed = time.time() - start_time

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f'\nDataset built in {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'   X shape       : {X.shape}')
    print(f'   y shape       : {y.shape}')
    print(f'   Classes found : {found_classes} / {len(target_karsl_classes)}')
    print(f'   Empty classes  : {empty_classes}')

    np.savez_compressed(NPZ_PATH, X=X, y=y)
    print(f'\nSaved: {NPZ_PATH}')
"""

# ===================================================================
# Apply patches
# ===================================================================
code_idx = 0
cell4_nb_idx = None
cell6_nb_idx = None

for i, cell in enumerate(nb['cells']):
    if cell.get("cell_type") == "code":
        code_idx += 1
        src = "".join(cell.get("source", []))
        if "CELL 4:" in src and "AUTO-DISCOVER" in src:
            cell4_nb_idx = i
        elif "CELL 6: BUILD DATASET" in src:
            cell6_nb_idx = i

print(f"Cell 4 (vocab) at nb index: {cell4_nb_idx}")
print(f"Cell 6 (dataset) at nb index: {cell6_nb_idx}")

if cell4_nb_idx is not None:
    nb['cells'][cell4_nb_idx]['source'] = [new_cell4]
    nb['cells'][cell4_nb_idx]['outputs'] = []
    print("  Patched Cell 4 — nested folder auto-detection")

if cell6_nb_idx is not None:
    nb['cells'][cell6_nb_idx]['source'] = [new_cell6]
    nb['cells'][cell6_nb_idx]['outputs'] = []
    print("  Patched Cell 6 — recursive file scanning (rglob)")

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved updated notebook!")
print("The notebook now correctly handles the nested KArSL structure:")
print("  KARSL-502/01/01/train/*.mp4")
print("  KARSL-502/01/01/test/*.mp4")

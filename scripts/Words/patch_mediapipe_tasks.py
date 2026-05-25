import json

NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Keypoints_Training_Kaggle.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 3: update KARSL_ROOT path ──────────────────────────────
new_cell3 = r"""# CELL 3: CONFIGURATION

IS_KAGGLE  = os.path.exists('/kaggle')
OUTPUT_DIR = Path('/kaggle/working') if IS_KAGGLE else Path(r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) if not IS_KAGGLE else None

# ── Hyper-parameters ──────────────────────────────────────────────
SEQUENCE_LENGTH = 48
BATCH_SIZE      = 64
EPOCHS          = 150
LEARNING_RATE   = 5e-4
LSTM_UNITS_1    = 256
LSTM_UNITS_2    = 128
LSTM_UNITS_3    = 64
DENSE_UNITS     = 256
DROPOUT_RATE    = 0.4
TEST_SIZE       = 0.4

# ── Hardcoded paths ───────────────────────────────────────────────
if IS_KAGGLE:
    KARSL_ROOT  = Path('/kaggle/input/blablabla/karsl-502')
    LABELS_FILE = '/kaggle/input/datasets/ahmed171102/karsl-502-labels/KARSL-502_Labels.txt'
else:
    KARSL_ROOT  = Path(r'M:\Term 10\Grad\SLR Main\Words\Datasets\KArSL_502')
    LABELS_FILE = str(Path(r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\KARSL-502_Labels.txt'))

# ── Verify ────────────────────────────────────────────────────────
if not KARSL_ROOT.exists():
    raise FileNotFoundError(f'Dataset not found: {KARSL_ROOT}')

print(f'KArSL root   : {KARSL_ROOT}  OK')
print(f'Labels file  : {LABELS_FILE}  {"OK" if os.path.exists(LABELS_FILE) else "NOT FOUND"}')
print(f'Output dir   : {OUTPUT_DIR}')
print(f'Sequence len : {SEQUENCE_LENGTH}')
print(f'Running on   : {"Kaggle" if IS_KAGGLE else "Local"}')
"""

# ── Cell 5: fix structure — NO double signer folder ──────────────
new_cell5 = r"""# CELL 5: BUILD RECORDING MAP
# Structure: KARSL_ROOT/{signer}/{split}/{class_id}/lh_keypoints/
# e.g. /kaggle/input/blablabla/karsl-502/01/test/0289/lh_keypoints/

print('=' * 60)
print('BUILDING RECORDING MAP')
print('=' * 60)

class_recordings = {}
FEATURE_DIM = None

top_entries = sorted([e.name for e in os.scandir(str(KARSL_ROOT)) if e.is_dir()])
print(f'Signer folders: {top_entries}')

for signer in top_entries:
    signer_path = KARSL_ROOT / signer   # e.g. 01/  (no double folder)

    for split in ['train', 'test']:
        split_path = signer_path / split
        if not split_path.exists():
            continue

        for cls_entry in os.scandir(str(split_path)):
            if not cls_entry.is_dir() or not cls_entry.name.isdigit():
                continue

            class_id = int(cls_entry.name)
            lh_dir   = Path(cls_entry.path) / 'lh_keypoints'
            rh_dir   = Path(cls_entry.path) / 'rh_keypoints'

            if not lh_dir.exists() or not rh_dir.exists():
                continue

            lh_map = {Path(p.path).stem: p.path for p in os.scandir(str(lh_dir)) if p.name.endswith('.npy')}
            rh_map = {Path(p.path).stem: p.path for p in os.scandir(str(rh_dir)) if p.name.endswith('.npy')}
            common = set(lh_map) & set(rh_map)
            if not common:
                continue

            if class_id not in class_recordings:
                class_recordings[class_id] = []
            for stem in sorted(common):
                class_recordings[class_id].append((lh_map[stem], rh_map[stem]))
                if FEATURE_DIM is None:
                    try:
                        arr = np.load(lh_map[stem])
                        FEATURE_DIM = arr.shape[-1] if arr.ndim >= 2 else len(arr)
                    except:
                        pass

if not class_recordings:
    print('\nERROR: No lh_keypoints / rh_keypoints found!')
    raise FileNotFoundError('No .npy keypoint files found.')

class_ids    = sorted(class_recordings.keys())
NUM_FEATURES = (FEATURE_DIM or 21) * 2
for cid in class_ids:
    id_to_english.setdefault(cid, str(cid))
    id_to_arabic.setdefault(cid, str(cid))

total_recs = sum(len(v) for v in class_recordings.values())
named      = sum(1 for c in class_ids if id_to_english[c] != str(c))

print(f'\nClasses found  : {len(class_ids)} ({named} with named labels)')
print(f'Total rec pairs: {total_recs}')
print(f'Avg per class  : {total_recs / max(len(class_ids),1):.1f}')
print(f'Feature dim    : {FEATURE_DIM} per hand -> {NUM_FEATURES} total')
print(f'\nFirst 10 classes:')
for cid in class_ids[:10]:
    print(f'  {cid:4d}  {id_to_english[cid]:25s}  {len(class_recordings[cid])} recordings')
"""

# Apply both patches
code_idx = 0
patched = []
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        code_idx += 1
        if code_idx == 3:
            nb['cells'][i]['source'] = [new_cell3]
            nb['cells'][i]['outputs'] = []
            patched.append(f'Cell 3 (KARSL_ROOT -> /kaggle/input/blablabla/karsl-502)')
        elif code_idx == 5:
            nb['cells'][i]['source'] = [new_cell5]
            nb['cells'][i]['outputs'] = []
            patched.append(f'Cell 5 (structure: signer/split/class_id — no double folder)')

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done!")
for p in patched:
    print(f"  {p}")

import json

NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle_Independent.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell4 = r"""# =========================
# CELL 4: DISCOVER CLASSES & LOAD LABEL NAMES
# =========================
# Reads KARSL-502_Labels.txt to map folder numbers to real word names.
# File format: SignID \t Sign-Arabic \t Sign-English
#
# On Kaggle: Add the labels file as a second dataset named "karsl-502-labels"
# It will be available at: /kaggle/input/karsl-502-labels/KARSL-502_Labels.txt

print('=' * 60)
print('DISCOVERING CLASSES & LOADING LABELS')
print('=' * 60)

# ---- Step 1: Find the labels file ----
LABELS_FILE = None

if IS_KAGGLE:
    KAGGLE_INPUT = Path('/kaggle/input')

    # Priority 1: Direct path if user named the dataset "karsl-502-labels"
    direct = KAGGLE_INPUT / 'karsl-502-labels' / 'KARSL-502_Labels.txt'
    if direct.exists():
        LABELS_FILE = str(direct)

    # Priority 2: Scan ALL datasets in /kaggle/input/ for any .txt with "label" in name
    if LABELS_FILE is None:
        for dataset_dir in os.scandir(str(KAGGLE_INPUT)):
            if not dataset_dir.is_dir():
                continue
            for f in os.scandir(dataset_dir.path):
                if f.is_file() and 'label' in f.name.lower() and f.name.endswith('.txt'):
                    LABELS_FILE = f.path
                    break
            if LABELS_FILE:
                break

    # Priority 3: Walk deeper (in case of nested folders)
    if LABELS_FILE is None:
        for root, dirs, files in os.walk(str(KAGGLE_INPUT)):
            for fname in files:
                if 'label' in fname.lower() and fname.endswith('.txt'):
                    LABELS_FILE = os.path.join(root, fname)
                    break
            if LABELS_FILE:
                break
else:
    # Local path
    local = Path(r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\KARSL-502_Labels.txt')
    if local.exists():
        LABELS_FILE = str(local)

print(f'Labels file: {LABELS_FILE if LABELS_FILE else "NOT FOUND (will use folder numbers)"}')

# ---- Step 2: Build ID -> label mappings ----
id_to_english = {}
id_to_arabic = {}

if LABELS_FILE:
    with open(LABELS_FILE, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('SignID'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    sign_id = int(parts[0])
                    arabic  = parts[1].strip()
                    english = parts[2].strip()
                    id_to_english[sign_id] = english if english and english not in ('?', '??') else str(sign_id)
                    id_to_arabic[sign_id]  = arabic  if arabic  and arabic  not in ('?', '??') else english
                except (ValueError, IndexError):
                    continue
    print(f'Loaded {len(id_to_english)} label mappings from file')
else:
    print('No labels file found. Folder numbers will be used as class names.')

# ---- Step 3: Scan KArSL dataset folders ----
if not KARSL_ROOT.exists():
    raise FileNotFoundError(f'KArSL dataset not found: {KARSL_ROOT}')

class_ids = sorted([
    int(e.name) for e in os.scandir(str(KARSL_ROOT))
    if e.is_dir() and e.name.isdigit()
])

# Fill in fallback labels for any class without a mapping
for cid in class_ids:
    if cid not in id_to_english:
        id_to_english[cid] = str(cid)
    if cid not in id_to_arabic:
        id_to_arabic[cid] = str(cid)

target_karsl_classes = class_ids

print(f'\nTotal class folders : {len(class_ids)}')
print(f'With real labels    : {sum(1 for c in class_ids if id_to_english[c] != str(c))}')
print(f'\nSample labels:')
for cid in class_ids[:15]:
    print(f'  Folder {cid:3d} -> "{id_to_english[cid]}"')
"""

# Find code cell 4 and replace
code_idx = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        code_idx += 1
        if code_idx == 4:
            nb['cells'][i]['source'] = [new_cell4]
            nb['cells'][i]['outputs'] = []
            print(f"Patched code cell 4 at notebook index {i+1}")
            break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done! Notebook updated.")
print()
print("On Kaggle, name your labels dataset exactly: karsl-502-labels")
print("The notebook will find it at: /kaggle/input/karsl-502-labels/KARSL-502_Labels.txt")
print("If the name is different, it still scans all datasets as fallback.")

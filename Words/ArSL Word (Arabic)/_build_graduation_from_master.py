"""Build graduation vocabulary CSV from master KARSL-502 labels."""
import pandas as pd
from pathlib import Path

OUT = Path(r'M:/Term 10/Grad/SLR Main/Words/ArSL Word (Arabic)')

# ── 1. Load authoritative master labels (txt preferred, csv fallback) ───────
def load_master_labels():
    txt_path = OUT / 'KARSL-502_Labels.txt'
    csv_path = OUT / 'KARSL-502_Labels.csv'
    rows = []

    if txt_path.exists():
        with open(txt_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('SignID'):
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                try:
                    sign_id = int(parts[0])
                except ValueError:
                    continue
                arabic = parts[1].strip()
                english = parts[2].strip()
                rows.append({
                    'sign_id': sign_id,
                    'class_id': sign_id + 1,
                    'arabic': arabic,
                    'english': english,
                })
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            sign_id = int(r['SignID'])
            rows.append({
                'sign_id': sign_id,
                'class_id': sign_id + 1,
                'arabic': str(r['Sign-Arabic']).strip(),
                'english': str(r['Sign-English']).strip(),
            })
    else:
        raise FileNotFoundError('No KARSL-502_Labels.txt or .csv found')

    master = pd.DataFrame(rows)
    master['is_number'] = master['sign_id'].between(1, 31)
    master['is_letter'] = master['sign_id'].between(32, 70)
    master['is_word'] = ~(master['is_number'] | master['is_letter'])
    return master

master = load_master_labels()
print(f'Master labels loaded: {len(master)} entries')
print(f'  Numbers : {master["is_number"].sum()}')
print(f'  Letters : {master["is_letter"].sum()}')
print(f'  Words   : {master["is_word"].sum()}')

# ── 2. Graduation selection — sign_ids only (category for grouping) ───────────
#    class_id = sign_id + 1  (KArSL dataset folder number)
SELECTION = [
    # numbers (SignID 1–31)
    *[(i, 'number') for i in range(1, 32)],
    # greeting
    (289, 'greeting'), (290, 'greeting'), (291, 'greeting'), (292, 'greeting'),
    (293, 'greeting'), (294, 'greeting'), (295, 'greeting'), (297, 'greeting'),
    (298, 'greeting'), (375, 'greeting'),
    # family
    (192, 'family'), (193, 'family'), (194, 'family'), (195, 'family'),
    (196, 'family'), (197, 'family'), (198, 'family'), (199, 'family'),
    (200, 'family'), (209, 'family'), (210, 'family'), (216, 'family'),
    (217, 'family'), (223, 'family'),
    # verb
    (160, 'verb'), (161, 'verb'), (162, 'verb'), (163, 'verb'), (164, 'verb'),
    (165, 'verb'), (169, 'verb'), (170, 'verb'), (173, 'verb'), (174, 'verb'),
    (175, 'verb'), (181, 'verb'), (182, 'verb'), (185, 'verb'), (186, 'verb'),
    (189, 'verb'), (191, 'verb'),
    # adjective
    (224, 'adjective'), (226, 'adjective'), (227, 'adjective'), (229, 'adjective'),
    (231, 'adjective'), (272, 'adjective'),
    # emotion
    (234, 'emotion'), (235, 'emotion'), (237, 'emotion'), (238, 'emotion'),
    (239, 'emotion'), (240, 'emotion'), (247, 'emotion'), (254, 'emotion'),
    (255, 'emotion'), (256, 'emotion'), (265, 'emotion'),
    # direction
    (273, 'direction'), (275, 'direction'), (279, 'direction'), (281, 'direction'),
    (282, 'direction'), (283, 'direction'), (285, 'direction'), (286, 'direction'),
    (287, 'direction'), (288, 'direction'),
    # home
    (299, 'home'), (302, 'home'), (303, 'home'), (304, 'home'), (306, 'home'),
    (312, 'home'), (321, 'home'), (325, 'home'), (328, 'home'), (336, 'home'),
    (341, 'home'), (346, 'home'),
    # health
    (92, 'health'), (100, 'health'), (113, 'health'), (115, 'health'),
    (116, 'health'), (117, 'health'), (132, 'health'), (134, 'health'),
    (497, 'health'), (498, 'health'),
    # profession
    (468, 'profession'), (484, 'profession'), (486, 'profession'), (491, 'profession'),
    (492, 'profession'), (500, 'profession'),
    # people
    (202, 'people'), (203, 'people'), (204, 'people'), (218, 'people'),
]

sel = pd.DataFrame(SELECTION, columns=['sign_id', 'category'])
sel = sel.drop_duplicates('sign_id').sort_values('sign_id')

# ── 3. Join with master — fail on unknown sign_ids ───────────────────────────
merged = sel.merge(master, on='sign_id', how='left')
missing = merged[merged['english'].isna()]
if len(missing):
    print('ERROR: sign_ids not in master labels:', missing['sign_id'].tolist())
    raise SystemExit(1)

# Reorder columns for human-readable export
out_cols = ['sign_id', 'class_id', 'english', 'arabic', 'category']
graduation = merged[out_cols].sort_values(['category', 'sign_id'])

# Selection-only file (edit this to add/remove words)
selection_only = graduation[['sign_id', 'category']].copy()

selection_path = OUT / 'KARSL-502_GraduationSelection.csv'
full_path = OUT / 'KARSL-502_BasicWords.csv'

selection_only.to_csv(selection_path, index=False)
graduation.to_csv(full_path, index=False)

print(f'\nGraduation selection : {len(selection_only)} sign_ids')
print(selection_only.groupby('category').size().to_string())
print(f'\nWrote selection : {selection_path.name}  (edit this — sign_id + category only)')
print(f'Wrote enriched  : {full_path.name}  (auto-built from master labels)')
print('\nSample from master join:')
print(graduation.head(8).to_string(index=False))

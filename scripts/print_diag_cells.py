import json, sys
sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main\Letters_ORIGINAL\Base_Pipeline_English_Letters\SLR_Diagnostics.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print("Total cells:", len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    print(f"Cell {i} ({ctype}): {src[:80].strip()}")

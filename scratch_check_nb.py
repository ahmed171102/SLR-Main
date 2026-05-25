import json

file_path = r'm:\Term 10\Grad\SLR Main\Unified_Dataset_Merger.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"Cell {i} ({cell['cell_type']})")
    if cell['source']:
        print(f"  First line: {repr(cell['source'][0])}")

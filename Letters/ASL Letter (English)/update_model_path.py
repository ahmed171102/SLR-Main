import json
import os

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_English.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Search for the cell containing MODEL_PATH
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'MODEL_PATH = ' in line:
                source[i] = 'MODEL_PATH = os.path.join("Base_Pipeline_English_Letters", "asl_mediapipe_mlp_model.h5")\n'
                print(f"Updated line: {line.strip()} -> {source[i].strip()}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

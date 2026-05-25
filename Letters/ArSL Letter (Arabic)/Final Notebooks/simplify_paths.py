import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\Production_Architecture_Arabic.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'MLP_MODEL_PATH = ' in line:
                source[i] = 'MLP_MODEL_PATH = r"arsl_mediapipe_mlp_model_final.h5"\n'
            if 'MOBILENET_MODEL_PATH = ' in line:
                source[i] = 'MOBILENET_MODEL_PATH = r"mobilenet_arabic_final.h5"\n'
            if 'DATASET_PATH = ' in line:
                source[i] = 'DATASET_PATH = r"FINAL_CLEAN_DATASET.csv"\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Paths simplified to use local directory.")

import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\Production_Architecture_Arabic.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Check for the misaligned lines in extract_features
            if 'if results.multi_hand_landmarks:' in line and 'def extract_features' in "".join(source):
                if line.startswith('            if'):
                    source[i] = '    if results.multi_hand_landmarks:\n'
            
            if 'landmarks = results.multi_hand_landmarks[0].landmark' in line:
                if line.startswith('        landmarks'):
                    source[i] = '        landmarks = results.multi_hand_landmarks[0].landmark\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Indentation fixed in Arabic notebook.")

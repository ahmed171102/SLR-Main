import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ArSL Letter (Arabic)\Production_Architecture_Arabic.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Fix the error message that uses the undefined MODEL_PATH
            if '{MODEL_PATH}' in line:
                source[i] = line.replace('{MODEL_PATH}', '{MLP_MODEL_PATH}')
            
            # Fix indentation for the hand drawing if
            if line.strip() == 'if results.multi_hand_landmarks:':
                 if line.startswith('    if'): # 4 spaces - incorrect
                     source[i] = '            if results.multi_hand_landmarks:\n'
            
            # Ensure the loop inside hand drawing is also indented correctly
            if 'for hand_landmarks in results.multi_hand_landmarks:' in line:
                 if not line.startswith('                for'):
                     source[i] = '                for hand_landmarks in results.multi_hand_landmarks:\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Arabic Notebook fixed: MODEL_PATH reference removed and indentation corrected.")

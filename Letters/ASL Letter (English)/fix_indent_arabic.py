import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_Arabic.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if line.strip() == 'if results.multi_hand_landmarks:':
                if line.startswith('        if'): 
                    source[i] = '            if results.multi_hand_landmarks:\n'
            if 'for hand_landmarks in results.multi_hand_landmarks:' in line:
                 if not line.startswith('                for'):
                     source[i] = '                for hand_landmarks in results.multi_hand_landmarks:\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Arabic Indentation fixed.")

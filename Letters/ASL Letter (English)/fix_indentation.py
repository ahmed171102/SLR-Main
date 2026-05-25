import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_English.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Specific fix for the misaligned 'if results.multi_hand_landmarks:'
            if line.strip() == 'if results.multi_hand_landmarks:':
                if line.startswith('        if'): # 8 spaces
                    source[i] = '            if results.multi_hand_landmarks:\n'
                    print(f"Fixed indentation at line {i}")
            
            # Also check the next line just in case
            if 'for hand_landmarks in results.multi_hand_landmarks:' in line:
                 if not line.startswith('                for'):
                     source[i] = '                for hand_landmarks in results.multi_hand_landmarks:\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Indentation fixed.")

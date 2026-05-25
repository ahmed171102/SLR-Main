import json

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Base_Pipeline_English_Letters\MobileNetV2_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # Look for the hardcoded class_labels list in the camera block
        if "class_labels = ['A', 'B'," in source:
            new_source = []
            for line in cell['source']:
                if "class_labels = ['A', 'B'," in line or \
                   "'K', 'L', " in line or \
                   "'U', 'V', " in line or \
                   "# Correct class labels" in line:
                    continue # remove these lines
                new_source.append(line)
            
            # Insert a note
            new_source.insert(4, "    # Note: We are reusing the dynamically generated `class_labels` dictionary from earlier.\n")
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("Camera block patched successfully!")

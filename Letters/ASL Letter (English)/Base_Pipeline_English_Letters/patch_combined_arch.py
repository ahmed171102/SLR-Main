import json

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Base_Pipeline_English_Letters\Combined_Architecture.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # 1. Remove the hardcoded class list
        if "class_labels = ['A', 'B', 'C', 'D'" in source:
            new_source = []
            for line in cell['source']:
                if "class_labels =" in line or "'K', 'L', 'M'" in line or "'U', 'V', 'W'" in line or "# Class labels" in line:
                    continue
                new_source.append(line)
            # Add a note replacing it
            new_source.append("\n# Note: MobileNet uses the same alphabetical labels as MediaPipe, so we reuse the `encoder.classes_` generated dynamically from the CSV.\n")
            cell['source'] = new_source
            
        # 2. Update the prediction logic to use the encoder instead of the hardcoded array
        if "mob_label = class_labels[mob_idx]" in source:
            cell['source'] = source.replace("mob_label = class_labels[mob_idx]", "mob_label = encoder.inverse_transform([mob_idx])[0]").splitlines(keepends=True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("Combined_Architecture.ipynb patched successfully!")

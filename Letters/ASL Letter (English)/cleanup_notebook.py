import json

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_English.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        cleaned_source = []
        for line in source:
            if 'mp_drawing_styles.get_default_face_mesh_tesselation_style())' in line:
                continue
            if 'mp_drawing_styles.get_default_pose_landmarks_style())' in line:
                continue
            if '# Draw Holistic landmarks' in line:
                line = '            # Draw Hand landmarks\n'
            cleaned_source.append(line)
        cell['source'] = cleaned_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cleanup completed.")

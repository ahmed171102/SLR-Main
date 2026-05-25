import json

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\MobileNetV2_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = "".join(cell.get('source', []))
        if "cv2.COLOR_BGR2RGBRGB" in src:
            new_source = []
            for line in cell['source']:
                if "cv2.COLOR_BGR2RGBRGB" in line:
                    line = line.replace("cv2.COLOR_BGR2RGBRGB", "cv2.COLOR_BGR2RGB")
                new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("Typo fixed!")

import json

file_path = r"m:\Term 10\Grad\SLR Main\Words\ASL Word (English)\ASL_Word_Live_Test (1).ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        # Check if we need to fix it
        fixed = False
        for i, line in enumerate(source):
            if "pred_idx = np.argmax(proba)" in line:
                # Only insert if it's not already there
                if len(source) > i+1 and "pred_conf" not in source[i+1]:
                    source.insert(i+1, "                pred_conf = proba[pred_idx]\n")
                    source.insert(i+2, "                pred_word = index_to_word.get(pred_idx, 'Unknown')\n")
                    fixed = True
                break
        
        # Also clear the outputs which show the error, so it's clean for the user
        if fixed:
            cell["outputs"] = []
            cell["execution_count"] = None

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed notebook.")

import json

path = r"m:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\Unified_Word_Training_Version2.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            # Change 1: File picker support for xlsx
            if 'filetypes=[("CSV Files", "*.csv")]' in line:
                line = line.replace('filetypes=[("CSV Files", "*.csv")]', 'filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")]')
            
            # Change 2: Label text
            if 'text="Vocab CSV (Optional for Arabic):"' in line:
                line = line.replace('text="Vocab CSV (Optional for Arabic):"', 'text="Vocab File (Optional for Arabic):"')
            
            # Change 3: Print statement
            if '# 1. If you selected a CSV in the popup:' in line:
                line = line.replace('CSV in the popup', 'file in the popup')
            if 'Loading Vocab from CSV:' in line:
                line = line.replace('from CSV', 'from file')
            
            # Change 4: Reading logic
            if "vocab = pd.read_csv(VOCAB_CSV, encoding='utf-8-sig', header=None, dtype=str)" in line:
                new_source.extend([
                    "    if str(VOCAB_CSV).lower().endswith(('.xlsx', '.xls')):\n",
                    "        vocab = pd.read_excel(VOCAB_CSV, header=None, dtype=str)\n",
                    "    else:\n",
                    "        vocab = pd.read_csv(VOCAB_CSV, encoding='utf-8-sig', header=None, dtype=str)\n"
                ])
                continue
                
            new_source.append(line)
        cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    # add trailing newline to match jupyter standard
    f.write("\n")

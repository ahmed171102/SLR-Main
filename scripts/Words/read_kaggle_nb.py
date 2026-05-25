import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\kaleem-app.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    ctype = cell.get("cell_type", "?")
    print(f"\n=== Cell {i+1} [{ctype}] ===")
    print(src[:2000])  # first 2000 chars per cell

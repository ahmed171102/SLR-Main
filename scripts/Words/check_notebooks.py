import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_notebook(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    p = Path(path)
    if not p.exists():
        print("  [MISSING] FILE DOES NOT EXIST")
        return

    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"  Total cells: {len(cells)}")

    errors = []
    last_outputs = []

    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        first_line = src.strip().split("\n")[0][:100] if src.strip() else "(empty)"
        outputs = cell.get("outputs", [])
        cell_type = cell.get("cell_type", "?")

        has_error = False
        cell_out = []
        for o in outputs:
            otype = o.get("output_type", "")
            if otype == "error":
                has_error = True
                ename = o.get("ename", "")
                evalue = o.get("evalue", "")
                errors.append(f"Cell {i+1}: {ename} - {evalue}")
                cell_out.append(f"  [ERR] ERROR: {ename}: {evalue}")
            elif otype == "stream":
                txt = "".join(o.get("text", []))[:200]
                cell_out.append(f"  OUT: {txt.strip()}")

        status = "[ERR]" if has_error else ("[OK] " if outputs else "[   ]")
        print(f"  Cell {i+1:02d} [{cell_type:8s}] {status}  {first_line}")
        if cell_out:
            for line in cell_out:
                print(f"         {line}")

    print()
    if errors:
        print(f"  [!!] ERRORS FOUND ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  [OK] No errors found in last run outputs.")

# ---- Check Unified notebook ----
check_notebook(
    r"M:\Term 10\Grad\SLR Main\Words\ASL Word (English)\Unified_Word_Training_Version2.ipynb",
    "UNIFIED WORD TRAINING (Version 2)"
)

# ---- Check ArSL Training notebook ----
check_notebook(
    r"M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training.ipynb",
    "ArSL WORD TRAINING"
)

# ---- Check ArSL Kaggle notebook ----
check_notebook(
    r"M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle.ipynb",
    "ArSL WORD TRAINING (Kaggle)"
)

# ---- Check ArSL Live Test notebook ----
check_notebook(
    r"M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Live_Test.ipynb",
    "ArSL LIVE TEST"
)

# ---- Check ASL Live Test notebook ----
check_notebook(
    r"M:\Term 10\Grad\SLR Main\Words\ASL Word (English)\ASL_Word_Live_Test (1).ipynb",
    "ASL LIVE TEST (1)"
)

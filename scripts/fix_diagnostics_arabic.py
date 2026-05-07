"""
Fix SLR_Diagnostics.ipynb Cell 4 — update the Arabic config block to match
the production model: arsl_mediapipe_mlp_model_bestV2.2.h5 + ASLAD-3000_v2.csv (32 classes)
"""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

DIAG_PATH = (
    r"C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main"
    r"\Letters_ORIGINAL\Base_Pipeline_English_Letters\SLR_Diagnostics.ipynb"
)

MODEL_PATH_V2  = (
    r"C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main"
    r"\Letters_ORIGINAL\ArSL (Arabic Letters)\arsl_mediapipe_mlp_model_bestV2.2.h5"
)

DATASET_PATH_V2 = (
    r"C:\Users\HADEEL GAMALELDIN\Desktop\SLR-Main"
    r"\Letters_ORIGINAL\ArSL (Arabic Letters)\ASLAD-3000_v2.csv"
)

NEW_CELL4_SOURCE = f'''\
LANGUAGE = "arabic"   # <── change to "english" for ASL

if LANGUAGE == "english":
    MODEL_PATH   = r"asl_mediapipe_mlp_model.h5"
    DATASET_PATH = r"asl_mediapipe_keypoints_dataset.csv"
    CLASS_LABELS = [
        "A","B","C","D","E","F","G","H",
        "I","J","K","L","M","N","O","P",
        "Q","R","S","T","U","V","W","X",
        "Y","Z","del","nothing","space"
    ]
    TITLE_PREFIX = "ASL (English)"
else:
    # ── V2.2 Production model (ASLAD-3000_v2 dataset, 32 classes) ──────────────
    MODEL_PATH   = r"{MODEL_PATH_V2}"
    DATASET_PATH = r"{DATASET_PATH_V2}"
    CLASS_LABELS = [
        "ain",   "al",    "aleff", "bb",    "dal",   "dha",
        "dhad",  "fa",    "gaaf",  "ghain", "ha",    "haa",
        "jeem",  "kaaf",  "khaa",  "la",    "laam",  "meem",
        "nun",   "ra",    "saad",  "seen",  "sheen", "ta",
        "taa",   "thaa",  "thal",  "toot",  "waw",   "ya",
        "yaa",   "zay"
    ]
    TITLE_PREFIX = "ArSL V2.2 Production (32 classes)"

print(f"Language : {{LANGUAGE.upper()}}")
print(f"Model    : {{MODEL_PATH}}")
print(f"Dataset  : {{DATASET_PATH}}")
print(f"Classes  : {{len(CLASS_LABELS)}}")
'''

with open(DIAG_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Verify cell 4 is the config cell
cell4_src = ''.join(nb['cells'][4]['source'])
assert 'LANGUAGE' in cell4_src, f"Cell 4 is not the config cell! Content: {cell4_src[:100]}"

# Update source (notebook format: list of strings)
nb['cells'][4]['source'] = [line + '\n' for line in NEW_CELL4_SOURCE.splitlines()]
# Fix the last line (no trailing newline)
nb['cells'][4]['source'][-1] = nb['cells'][4]['source'][-1].rstrip('\n')

with open(DIAG_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("SUCCESS — Cell 4 updated.")
print("  LANGUAGE   : arabic")
print("  MODEL_PATH : arsl_mediapipe_mlp_model_bestV2.2.h5")
print("  DATASET    : ASLAD-3000_v2.csv")
print("  CLASSES    : 32 (ain → zay, matching production)")
print()
print("Next steps:")
print("  1. Open SLR_Diagnostics.ipynb in Jupyter")
print("  2. Run all cells (Kernel > Restart & Run All)")
print("  3. Read Cell 10 for overall accuracy, Cell 11 for per-class F1")

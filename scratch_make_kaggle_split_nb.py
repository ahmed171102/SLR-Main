import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    src = Path(r"M:/Term 10/Grad/SLR Main/Words/ASL Word (English)/Unified_Word_Training_Version2_split.ipynb")
    dst = Path(r"M:/Term 10/Grad/SLR Main/Words/ASL Word (English)/Unified_Word_Training_Version2_split_kaggle_v2.ipynb")

    nb = json.loads(src.read_text(encoding="utf-8"))
    nb2 = deepcopy(nb)

    # Ensure metadata.id exists (per your notebook format instructions)
    for cell in nb2.get("cells", []):
        md = cell.setdefault("metadata", {})
        if "id" not in md and "id" in cell:
            md["id"] = cell["id"]

    cells = nb2.get("cells", [])

    def find_cell_index(predicate):
        for i, c in enumerate(cells):
            if predicate(c):
                return i
        return None

    # 1) Replace the first code cell under "Cell 2: Global config"
    idx_cfg = find_cell_index(
        lambda c: c.get("cell_type") == "code"
        and c.get("source")
        and c["source"][0].startswith("# =========================")
        and any("CELL 2: GLOBAL CONFIG" in s for s in c["source"][:6])
    )
    if idx_cfg is None:
        raise RuntimeError("Couldn't locate global config code cell")

    cells[idx_cfg]["source"] = [
        "# =========================",
        "# CELL 2: GLOBAL CONFIG (KAGGLE + LOCAL, WITH AUTO-NUMBERING)",
        "# =========================",
        "# ---- choose language ----",
        "LANGUAGE = \"asl\"  # \"asl\" or \"arsl\"",
        "",
        "# ---- Kaggle detection ----",
        "IS_KAGGLE = Path('/kaggle/input').exists()",
        "if IS_KAGGLE:",
        "    KAGGLE_INPUT = Path('/kaggle/input')",
        "    KAGGLE_WORKING = Path('/kaggle/working')",
        "    KAGGLE_INPUT_DIRS = sorted([p for p in KAGGLE_INPUT.iterdir() if p.is_dir()])",
        "    print('✅ Kaggle detected')",
        "    print('Kaggle input dirs:', [p.name for p in KAGGLE_INPUT_DIRS])",
        "else:",
        "    KAGGLE_INPUT = None",
        "    KAGGLE_WORKING = None",
        "    KAGGLE_INPUT_DIRS = []",
        "    print('ℹ️ Kaggle not detected; using local paths')",
        "",
        "# ---- local fallback root path ----",
        "PROJECT_ROOT = Path(r'M:/Term 10/Grad')  # <- CHANGE THIS (LOCAL ONLY)",
        "WORDS_ROOT_LOCAL = PROJECT_ROOT / 'SLR Main/Words'",
        "WLASL_ROOT_LOCAL = PROJECT_ROOT / 'Words dataset'",
        "WORDS_DATASETS_ROOT_LOCAL = WORDS_ROOT_LOCAL / 'Datasets'",
        "",
        "def _first_existing(*candidates):",
        "    for p in candidates:",
        "",
        "def _find_wlasl_json_kaggle(search_roots):",
        "    # Fast checks for common layouts (no recursion)",
        "    common_json_rels = [",
        "        'WLASL_v0.3.json',",
        "        'Datasets/WLASL_v0.3.json',",
        "        'Words Datasets/WLASL_v0.3.json',",
        "    ]",
        "    for root in search_roots:",
        "        for rel in common_json_rels:",
        "            p = root / rel",
        "            if p.exists():",
        "                return p",
        "",
        "    # Fallback: walk but prune heavy video folders", 
        "    skip_dirs = {",
        "        'wlasl_videos', 'videos', 'video', 'mp4', 'frames', 'images',",
        "        'train', 'test', 'val', 'validation',",
        "    }",
        "    for root in search_roots:",
        "        for dirpath, dirnames, filenames in os.walk(root):",
        "            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]",
        "            if 'WLASL_v0.3.json' in filenames:",
        "                return Path(dirpath) / 'WLASL_v0.3.json'",
        "    return None",
        "",
        "def _pick_dir_with_mp4(folder: Path | None):",
        "    if folder is None:",
        "        return None",
        "    folder = Path(folder)",
        "    if not folder.exists() or not folder.is_dir():",
        "        return None",
        "    if next(folder.glob('*.mp4'), None) is not None:",
        "        return folder",
        "    # Common subfolders", 
        "    for name in ['WLASL2000', 'videos', 'Videos', 'mp4', 'MP4']:",
        "        sub = folder / name",
        "        if sub.exists() and sub.is_dir() and next(sub.glob('*.mp4'), None) is not None:",
        "            return sub",
        "    # If exactly one immediate child has mp4 directly, use it",
        "    for child in folder.iterdir():",
        "        if child.is_dir() and next(child.glob('*.mp4'), None) is not None:",
        "            return child",
        "    # Last resort: return as-is",
        "    return folder",
        "        if p is not None and Path(p).exists():",
        "            return Path(p)",
        "    return None",
        "    # Expect you uploaded a Kaggle Dataset that contains WLASL_v0.3.json and videos.",
        "    # This supports several common folder layouts without scanning every .mp4.",
        "    search_roots = KAGGLE_INPUT_DIRS if KAGGLE_INPUT_DIRS else [Path('/kaggle/input')]",
        "    wlasl_json = _find_wlasl_json_kaggle(search_roots)",
        "    if wlasl_json is None:",
        "        raise FileNotFoundError('Kaggle: could not find WLASL_v0.3.json under /kaggle/input/<your_dataset>/')",
        "",
        "    video_root_candidates = []",
        "    for root in search_roots:",
        "        video_root_candidates.extend([",
        "            root / 'WLASL_videos',",
        "            root / 'Words Datasets/WLASL_videos',",
        "            root / 'Datasets/WLASL_videos',",
        "            root / 'videos',",
        "            root / 'WLASL2000',",
        "        ])",
        "",
        "    videos_dir = None",
        "    for cand in video_root_candidates:",
        "        picked = _pick_dir_with_mp4(cand)",
        "        if picked is not None:",
        "            videos_dir = picked",
        "            break",
        "",
        "    if videos_dir is None:",
        "        raise FileNotFoundError('Kaggle: could not find a videos folder containing .mp4 files under /kaggle/input/<your_dataset>/')",
        "        raise FileNotFoundError('Kaggle: missing WLASL_v0.3.json under /kaggle/input/<your_dataset>/')",
        "    if videos_dir is None:",
        "        raise FileNotFoundError('Kaggle: missing WLASL_videos/ under /kaggle/input/<your_dataset>/')",
        "",
        "    # For Kaggle we write outputs to /kaggle/working",
        "    WORK_DIR_DEFAULT = Path('/kaggle/working') / f'{LANGUAGE}_word_training'",
        "    WORDS_ROOT = Path('/kaggle/input')  # used only for searching shared vocab (read-only)",
        "else:",
        "    WORDS_ROOT = WORDS_ROOT_LOCAL",
        "    wlasl_json = _first_existing(",
        "        WLASL_ROOT_LOCAL / 'WLASL_v0.3.json',",
        "        WORDS_DATASETS_ROOT_LOCAL / 'WLASL_v0.3.json',",
        "    )",
        "    videos_dir = _first_existing(",
        "        WLASL_ROOT_LOCAL / 'Words Datasets/WLASL_videos',",
        "        WORDS_DATASETS_ROOT_LOCAL / 'WLASL_videos',",
        "    )",
        "    if wlasl_json is None:",
        "        raise FileNotFoundError('Local: missing WLASL_v0.3.json (check PROJECT_ROOT)')",
        "    if videos_dir is None:",
        "        raise FileNotFoundError('Local: missing WLASL_videos folder (check PROJECT_ROOT)')",
        "",
        "    WORK_DIR_DEFAULT = WORDS_ROOT / 'ASL Word (English)'",
        "",
        "# ---- language-specific configuration ----",
        "CFG = {",
        "    'asl': {",
        "        'name': 'ASL (English)',",
        "        'work_dir': WORK_DIR_DEFAULT,",
        "        # Kaggle writes vocab to WORK_DIR; local uses the existing file in the repo",
        "        'vocab_csv': (WORK_DIR_DEFAULT / 'asl_word_vocabulary.csv') if IS_KAGGLE else (WORDS_ROOT / 'ASL Word (English)/asl_word_vocabulary.csv'),",
        "        'dataset_type': 'wlasl',",
        "        'wlasl_json': wlasl_json,",
        "        'videos_dir': videos_dir,",
        "        'sequence_len': 30,",
        "        'features_per_frame': 258,  # <-- Make sure this is 258 for holistic!",
        "    },",
        "}",
        "",
        "C = CFG[LANGUAGE]",
        "WORK_DIR = C['work_dir']",
        "WORK_DIR.mkdir(parents=True, exist_ok=True)",
        "",
    ]

    # 2) Patch vocab helper cell to also look for shared vocab in Kaggle input dataset roots
    idx_vocab = find_cell_index(
        lambda c: c.get("cell_type") == "code"
        and c.get("source")
        and c["source"][0].startswith(
            "# If per-language vocab CSV doesn't exist yet, generate it from the shared vocab file."
        )
    )
    if idx_vocab is None:
        raise RuntimeError("Couldn't locate vocab helper code cell")

    cells[idx_vocab]["source"] = [
        "# If per-language vocab CSV doesn't exist yet, generate it from the shared vocab file.",
        "vocab_path = Path(C['vocab_csv'])",
        "",
        "shared_candidates = [",
        "    WORDS_ROOT / 'Shared/shared_word_vocabulary.csv',",
        "    WORDS_ROOT / 'ASL Word (English)/shared_word_vocabulary.csv',",
        "    WORK_DIR / 'shared_word_vocabulary.csv',",
        "]",
        "",
        "# Kaggle: also check top-level of each /kaggle/input/<dataset>/ for shared_word_vocabulary.csv",
        "if 'IS_KAGGLE' in globals() and IS_KAGGLE:",
        "    for d in KAGGLE_INPUT_DIRS:",
        "        shared_candidates.extend([",
        "            d / 'shared_word_vocabulary.csv',",
        "            d / 'Shared/shared_word_vocabulary.csv',",
        "            d / 'ASL Word (English)/shared_word_vocabulary.csv',",
        "        ])",
        "",
        "shared_csv = next((p for p in shared_candidates if p.exists()), None)",
        "",
    ]

    if dst.exists():
        dst.unlink()
    dst.write_text(json.dumps(nb2, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Wrote Kaggle notebook: {dst}")


if __name__ == "__main__":
    main()

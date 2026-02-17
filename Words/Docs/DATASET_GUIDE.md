# Datasets Guide — WLASL & KArSL

> **For team members** — Where to get the data, what's downloaded, and how it's structured

---

## Dataset Status Summary

| Dataset | Language | Status | Size | Location |
|---|---|---|---|---|
| **WLASL** | English ASL | ✅ **Ready** (11,980 videos extracted) | ~5 GB | `Words/Datasets/WLASL_videos/` |
| **KArSL-502** | Arabic ArSL | ❌ **Not downloaded** | ~2-4 GB | Needs: `Words/Datasets/KArSL_502/` |

---

## 1. WLASL Dataset (English ASL Words)

### What Is It?
- **WLASL** = Word-Level American Sign Language
- 2,000 ASL word glosses with video clips from online sources
- Each video shows one person performing one sign
- ~2-10 seconds per video

### Download Link
- **Kaggle:** https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
- **Original paper:** https://dxli94.github.io/WLASL/

### Current State on Your Machine
```
E:\Term 9\Grad\Words dataset\                ← metadata files
    ├── WLASL_v0.3.json        (11.9 MB)    ← full gloss + video metadata
    ├── nslt_2000.json         (1.1 MB)     ← 2000-class train/val/test splits
    ├── nslt_1000.json         (704 KB)     ← 1000-class subset
    ├── nslt_300.json          (272 KB)     ← 300-class subset
    ├── nslt_100.json          (107 KB)     ← 100-class subset
    ├── missing.txt            (55 KB)      ← video IDs that are unavailable
    ├── wlasl_class_list.txt   (23 KB)      ← all 2001 class names
    └── archive (1).zip        (5.2 GB)     ← original download (can delete)

E:\Term 9\Grad\Main\...\SLR Main\Words\Datasets\WLASL_videos\
    ├── 00335.mp4
    ├── 00336.mp4
    ├── ...
    └── (11,980 total .mp4 files)           ← ✅ EXTRACTED & READY
```

### How We Filter It
We don't use all 2,000 classes. The notebook filters using `shared_word_vocabulary.csv`:
- 157 matched words that also exist in KArSL (Arabic)
- The `nslt_2000.json` split file maps `video_id → class_id`
- Only videos whose `class_id` matches one of our 157 words are used

---

## 2. KArSL Dataset (Arabic Sign Language Words)

### What Is It?
- **KArSL** = King Abdulaziz Sign Language (Saudi/Khaleeji dialect)
- 502 word classes performed by multiple signers
- Available as either raw videos or pre-extracted keypoints

### Download Links
| Source | Link |
|---|---|
| **Kaggle (primary)** | https://www.kaggle.com/datasets/yousefelkilany/karsl-502 |
| **Mendeley Data** | https://data.mendeley.com/datasets/y4b382tswr |
| **Alternative search** | Search "KArSL 502" on Kaggle |

### How to Set Up
1. Download from Kaggle (create account if needed)
2. Extract the archive
3. Place contents at:
```
SLR Main/Words/Datasets/KArSL_502/
    ├── 161/                    ← class folder (numbered by KArSL class ID)
    │   ├── sample1.npy         ← pre-extracted keypoints (preferred)
    │   ├── sample2.npy
    │   └── ...
    ├── 173/
    ├── 182/
    └── ...
```

### Supported File Formats
The ArSL notebook auto-detects the data format:

| Format | Extension | How It's Used |
|---|---|---|
| Pre-extracted keypoints | `.npy` | Loaded directly (fastest) |
| CSV keypoints | `.csv` | Read with pandas |
| Raw video | `.mp4` | Extracted via MediaPipe (slowest) |

Set `USE_PREEXTRACTED_KEYPOINTS = True` in the config cell if you have `.npy`/`.csv` files.  
Set `USE_PREEXTRACTED_KEYPOINTS = False` if you only have `.mp4` videos.

### Folder Name Patterns
The notebook tries these naming patterns for each class:
```python
KArSL_502/161/         # plain number
KArSL_502/161/         # 3-digit padded
KArSL_502/0161/        # 4-digit padded
KArSL_502/class_161/   # prefixed
```

---

## 3. Shared Vocabulary

Both datasets are linked through `Words/Shared/shared_word_vocabulary.csv`:

- **157 words** that exist in BOTH WLASL and KArSL
- Each word has a unique `word_id` (0-156)
- Contains: `wlasl_class` (English class ID) ↔ `karsl_class` (Arabic class ID)
- 9 categories: verb, family, adjective, object, health, direction, job, social, religion

### How the Filter Works
```
WLASL has 2,000 classes  ──┐
                            ├──→ shared_word_vocabulary.csv (157 matched words)
KArSL has 502 classes   ──┘

ASL notebook: "Give me only videos whose WLASL class is in the CSV"
ArSL notebook: "Give me only folders whose KArSL class is in the CSV"
```

---

## 4. Data Statistics

### WLASL (after filtering to 157 words)
- Total videos available: 11,980 (unfiltered)
- Matched videos: depends on WLASL_v0.3.json entries per class
- Expected: ~5-30 videos per word (varies by class)
- Some words may have very few samples (<5) — flagged in Data Exploration cell

### KArSL (expected, once downloaded)
- 502 total classes
- 157 used (matched to English)
- Typically 30-50 samples per class
- Higher quality (controlled recording environment)

---

## 5. Tips for Team Members

### If you're setting up from scratch:
1. Clone/pull the repo
2. WLASL videos are already extracted — no action needed
3. Download KArSL from Kaggle → place at `datasets/KArSL_502/`
4. Run the notebook top-to-bottom

### If MediaPipe extraction is too slow:
- First run takes 30-60 min (processes all videos)
- After that, it loads from `.npz` cache instantly
- Don't delete the `.npz` files!

### If you get OOM (Out of Memory):
- Reduce `BATCH_SIZE` in the Config cell (try 16 or 8)
- Close other GPU-using apps
- Monitor with `nvidia-smi -l 1`

### Data integrity check:
```python
# Quick check in any Python terminal:
import numpy as np
data = np.load('asl_word_sequences.npz')
print(f"Samples: {data['X'].shape[0]}, Classes: {len(np.unique(data['y']))}")
```

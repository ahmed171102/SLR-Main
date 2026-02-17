# Word Building Guide (Letters → Words / Words → Sentences)

This repo already has strong **single-frame** building blocks (hand keypoints CSVs + letter classifiers). Turning that into **words** requires either:

- **A) Fingerspelling**: keep your existing letter model and add a small stream decoder (fastest).
- **B) True word-sign recognition**: build **temporal sequences** from videos (WLASL/NSLT), then train a sequence model.

---

## Why your current CSV can’t directly become word sequences

In this workspace, the typical “Mediapipe keypoints CSV” for letters is **one row per frame**:

- **63 features** (21 hand landmarks × (x,y,z))
- plus a **label** column (the letter)

That format is great for **static letter classification**, but it has *no time dimension*. A word is a **sequence over time**, so you need either:

- a **decoder** that aggregates frame-by-frame letter predictions into a string (Approach A), or
- a **sequence dataset** where each training example is a **window of many frames** (Approach B).

---

## Approach A (fastest): Fingerspelling word builder using your letter model + decoder

Use your existing letter model (image-based or keypoints-based) to output a label per frame, then feed it into:

- [Guides/letter_stream_decoder.py](Guides/letter_stream_decoder.py)

### What this gives you

- Words from letters in real time: `H E L L O` → `HELLO`
- Handles `space`, `del`, and ignores `nothing`
- Reduces noisy frame flicker using a stability window + majority vote + cooldown

### Integration pseudocode (webcam loop)

```python
# Option 1: If you run this script from the same folder as letter_stream_decoder.py (Guides/)
# from letter_stream_decoder import LetterStreamDecoder

# Option 2: If you're in a notebook or running from elsewhere, add the Guides folder to sys.path
import sys
from pathlib import Path

GUIDES_DIR = Path(r"path/to/Sign_to_Sentence Project Main/Guides").resolve()
if str(GUIDES_DIR) not in sys.path:
   sys.path.insert(0, str(GUIDES_DIR))

from letter_stream_decoder import LetterStreamDecoder

decoder = LetterStreamDecoder(
    min_confidence=0.7,
    stable_window=5,
    majority_ratio=0.7,
    cooldown_s=0.6,
    control_labels=("space", "del", "nothing"),
)

while True:
    frame = webcam.read()

    # 1) Run your existing pipeline
    #    a) image model: label, conf = predict_from_rgb(frame)
    #    b) keypoints model: keypoints = mediapipe(frame); label, conf = predict_from_keypoints(keypoints)
    label, conf = model.predict(frame)

    # 2) Update decoder once per frame
    info = decoder.update(label, conf)   # returns dict

    # 3) Use info["text"] as your live sentence buffer
    overlay_text(frame, info["text"])  # show the built string

    # Optional: react to events
    if info["event"] == "append":
        print("Committed letter:", info["committed"], "Current word:", info["word"])
    elif info["event"] == "space":
        print("Space committed")
    elif info["event"] == "delete":
        print("Deleted last char")

    show(frame)
```

### Practical tips

- If you can’t type double letters (`LL`), you usually need the model to output a different stable label in between (often `nothing`) so the decoder can commit the same letter again later.
- You can tune:
  - `stable_window` bigger → more stable, but slower
  - `majority_ratio` bigger → stricter stability
  - `cooldown_s` bigger → fewer repeats

---

## Approach B: True word-sign recognition from WLASL (and optionally NSLT)

This is the “real” way to recognize **whole signs** like `HELLO`, `THANK YOU`, etc. The key difference: each training example is a **sequence** of frames.

### What you already have in the workspace root

- `WLASL_v0.3.json` (metadata: instance ids, glosses, urls, splits)
- `wlasl_class_list.txt` (class id ↔ gloss list)
- `missing.txt` (videos/instances that are unavailable)
- Optionally: `nslt_*.json` files (continuous sign language / sentence-level annotations)

### Step-by-step pipeline (WLASL)

1) **Parse WLASL metadata**
   - Read `WLASL_v0.3.json`
   - For each instance: get `video_id`, `url`, `start_time`, `end_time`, `gloss`, `split`

2) **Download / collect videos**
   - Download each instance url
   - Skip those listed in `missing.txt`
   - Keep a manifest of successes/failures

3) **Clip the signing segment (if timestamps exist)**
   - Trim videos to `[start_time, end_time]` to remove extra motion

4) **Extract per-frame keypoints**
   - For each frame (or every k-th frame), run MediaPipe
   - Save keypoints as sequences: `T × D`
     - Example: one hand only → `T × 63`
     - Both hands/pose/face → larger D (optional)

5) **Build sequence dataset**
   - Each sample becomes:
     - `X`: a tensor shaped `(T, D)` (pad/truncate to fixed T or use masking)
     - `y`: either a gloss class id (classification) or token sequence (CTC)

6) **Model options**

   **Option 1: Gloss classification (simpler)**
   - Input: `(T, D)`
   - Model: BiLSTM/GRU, Temporal CNN, or Transformer encoder
   - Output: softmax over gloss classes

   **Option 2: Continuous recognition (harder, closer to sentences)**
   - If using NSLT-like annotations:
   - Model: Transformer / BiLSTM + **CTC** (or seq2seq)
   - Output: token sequence (glosses) over time

7) **Evaluation metrics**

   For **word/gloss classification**:
   - Top-1 accuracy
   - Top-5 accuracy
   - Per-class F1 (macro) to handle imbalance

   For **continuous / sentence-level**:
   - WER (word error rate) over gloss tokens
   - CER (character error rate) if you convert glosses to text

### Notes on NSLT JSON (optional)

Your `nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json` can be used to move toward **continuous sign language** (sentences). The workflow is similar, but labels are **sequences** instead of single-class gloss ids.

---

## Dataset suggestions (letters + words)

Letters (fingerspelling):
- **ASL Alphabet** (static handshape images; good baseline)
- **Your repo’s Arabic letters dataset** (ArSL Letter folder) for Arabic fingerspelling

Words / Glosses:
- **WLASL** (large vocabulary isolated signs; fits Approach B)
- **MS-ASL** (another large isolated-sign dataset)

Continuous sign language (sentences):
- **RWTH-PHOENIX-Weather** (continuous sign recognition; very common in research)
- NSLT subsets (as available in your JSON files)

---

## 1–2 month study roadmap (practical)

Week 1–2: Foundations
- MediaPipe landmarks: coordinate normalization, missing frames
- Sequence basics: padding/masking, sliding windows, frame sampling
- Metrics: accuracy vs F1, WER for sequences

Week 3–4: Fingerspelling baseline (deliver something working)
- Integrate letter model + decoder (Approach A)
- Tune stability params; add `space/del` controls
- Record a small evaluation set (your own webcam recordings) and measure character accuracy

Week 5–8: Word-sign recognition (research-grade direction)
- Build WLASL downloader + manifest; handle missing
- Extract and store keypoint sequences efficiently
- Train a BiLSTM/GRU baseline (classification)
- If aiming for sentences: explore Transformer + CTC and evaluate with WER

---

## Quick decision rule

- If you need a demo **this week** → do **Approach A**.
- If your goal is “real” word/sign recognition for many signs → do **Approach B** (WLASL/NSLT).

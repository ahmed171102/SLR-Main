# Letters + Words Integration — Combined System Design

> **How letters and words will work together in real-time**

---

## The Big Picture: "My name is Ahmed"

```
User signs:              System recognizes:         Output built:
─────────────────────────────────────────────────────────────────────
[word sign: "my"]     →  Word Model → word_id=X   → "my"
[word sign: "name"]   →  Word Model → word_id=X   → "my name"
                         (pause — switch to letters)
[letter: A]           →  Letter Model → "A"       → "my name A"
[letter: H]           →  Letter Model → "H"       → "my name AH"
[letter: M]           →  Letter Model → "M"       → "my name AHM"
[letter: E]           →  Letter Model → "E"       → "my name AHME"
[letter: D]           →  Letter Model → "D"       → "my name AHMED"
                         (pause — back to words)
[word sign: "help"]   →  Word Model → word_id=2   → "my name AHMED help"
```

---

## Architecture: Dual-Model Real-Time System

```
                    ┌──────────────────────────────┐
                    │         WEBCAM FEED            │
                    │    (30 FPS continuous)          │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │      MediaPipe Hand Detection   │
                    │   21 landmarks × 3 = 63 features│
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │       MODE DETECTOR             │
                    │  "Is the hand moving or still?"  │
                    │                                  │
                    │  Still hand → LETTER MODE        │
                    │  Moving hand → WORD MODE         │
                    │  No hand → IDLE (space/pause)    │
                    └────────┬──────────┬─────────────┘
                             │          │
              ┌──────────────▼──┐  ┌────▼──────────────────┐
              │  LETTER MODEL   │  │    WORD MODEL          │
              │  (MLP)          │  │    (BiLSTM)            │
              │  Input: (1, 63) │  │  Input: (30, 63)       │
              │  Single frame   │  │  30-frame sequence     │
              │  ~23K params    │  │  ~320K params          │
              └────────┬────────┘  └────────┬───────────────┘
                       │                    │
              ┌────────▼────────────────────▼───────────────┐
              │           SENTENCE BUILDER                   │
              │  Commit-once-then-wait (letters)             │
              │  Cooldown + majority vote (words)            │
              └──────────────────┬───────────────────────────┘
                                 │
              ┌──────────────────▼───────────────────────────┐
              │              DISPLAY OUTPUT                   │
              │  English: "my name AHMED help"               │
              │  Arabic:  "اسمي أحمد يساعد"                  │
              └──────────────────────────────────────────────┘
```

---

## Key Differences: Letters vs. Words

| Feature | Letters | Words |
|---|---|---|
| **Model** | MLP (Dense) | BiLSTM (Recurrent) |
| **Input** | Single frame (1, 63) | 30-frame sequence (30, 63) |
| **What it detects** | Static hand shape | Hand shape changing over time |
| **Classes** | 29 (English) / 31 (Arabic) | 157 (bilingual) |
| **Training data** | Images | Video clips (2–10 sec) |
| **Inference speed** | Very fast (~1ms) | Slower (~10ms) |
| **Repetition control** | Commit-once-then-wait | Cooldown-based |

---

## Mode Detection Options

**Option A: Motion-Based (Recommended)**
- Track landmark movement between frames
- High movement over 30 frames → WORD mode
- Low movement / static pose → LETTER mode
- Threshold: `np.mean(np.abs(current - previous))` > `MOTION_THRESHOLD`

**Option B: Explicit Gesture Toggle**
- User makes a specific "switch" gesture (e.g., open/close fist)
- Simple but requires learning the toggle gesture

**Option C: Run Both Models**
- Run letter MLP on every frame AND word BiLSTM on rolling 30-frame window
- Use whichever has higher confidence
- More CPU usage but seamless switching

---

## What Already Exists vs. What Needs Building

**✅ Already Done (Letters):**
- MLP models trained (ASL + ArSL)
- MobileNetV2 models trained (ASL + ArSL)
- Combined fusion notebooks (both languages)
- Commit-once-then-wait inference strategy
- Letter Stream Decoder utility (`letter_stream_decoder.py`)
- Arabic display utilities (RTL text rendering)

**✅ Already Done (Words):**
- BiLSTM architecture defined
- WLASL dataset downloaded (11,980 videos)
- Shared vocabulary (157 bilingual words)
- Word training notebooks (ASL + ArSL)

**🔨 Needs Building:**
- Mode Detector (motion-based letter/word switching)
- Rolling 30-frame buffer for word model
- Combined letter + word webcam loop
- Sentence Builder (merge outputs from both models)
- Bilingual display (English + Arabic side by side)

---

## Shared Vocabulary Bridge

The Words module uses a **shared `word_id`** system:

```
shared_word_vocabulary.csv (in Words/Shared/):
┌─────────┬──────────┬──────────┬─────────────┬─────────────┐
│ word_id  │ english  │ arabic   │ wlasl_class │ karsl_class │
├─────────┼──────────┼──────────┼─────────────┼─────────────┤
│    0    │  drink   │  يشرب    │      1      │     161     │
│    1    │  chair   │  كرسي    │      4      │     328     │
│   ...   │   ...    │   ...    │    ...      │    ...      │
│   156   │  forgive │  يغفر   │    1753     │     446     │
└─────────┴──────────┴──────────┴─────────────┴─────────────┘

157 words across 9 categories
```

Both ASL and ArSL word models output the same `word_id`, allowing:
- English sign → word_id=0 → show "drink / يشرب"
- Arabic sign → word_id=0 → show "drink / يشرب"

Letters don't need this mapping — each language has its own alphabet.

---

## Pseudocode: Combined Inference Loop

```python
letter_model = load('asl_mediapipe_mlp_model_best.h5')
word_model = load('asl_word_lstm_model_best.h5')
frame_buffer = deque(maxlen=30)
sentence = ""
mode = "IDLE"

while webcam.isOpened():
    landmarks = mediapipe.extract(frame)
    frame_buffer.append(landmarks)
    
    # Movement detection
    if len(frame_buffer) >= 2:
        movement = np.mean(np.abs(landmarks - frame_buffer[-2]))
    else:
        movement = 0
    
    if not hand_detected:
        mode = "IDLE"
    elif movement > MOTION_THRESHOLD and len(frame_buffer) == 30:
        mode = "WORD"
    else:
        mode = "LETTER"
    
    if mode == "WORD":
        # BiLSTM on 30-frame sequence
        sequence = np.array(frame_buffer)  # shape: (30, 63)
        word_pred = word_model.predict(sequence[None, ...])
        if confident:
            sentence += word
            frame_buffer.clear()
    elif mode == "LETTER":
        # MLP on single frame
        letter_pred = letter_model.predict(landmarks[None, ...])
        committed = letter_decoder.feed(letter)
        if committed:
            sentence += committed
```

---

## Folder Structure Overview

```
SLR Main/
├── Letters/              ← Single-frame letter recognition
│   ├── ASL Letter (English)/      29 classes, MLP + MobileNet
│   ├── ArSL Letter (Arabic)/      31 classes, MLP + MobileNet
│   ├── Datasets/                  Raw images
│   ├── Arabic guide/              Helper scripts
│   ├── Guides/                    Reference implementations
│   ├── Orignal Notebooks/         Backup copies
│   └── Docs/                      ← you are here
│
└── Words/                ← 30-frame word recognition
    ├── ASL Word (English)/        157 classes, BiLSTM
    ├── ArSL Word (Arabic)/        157 classes, BiLSTM
    ├── Shared/                    shared_word_vocabulary.csv
    ├── Datasets/                  WLASL videos + KArSL
    └── Docs/                      Word module docs
```

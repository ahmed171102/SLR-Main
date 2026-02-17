# Letters + Words Integration — Combined System Design

> **How letters and words will work together in real-time**

---

## The Big Picture: "My name is Ahmed"

Here's exactly how a user would sign this sentence using **both** the letter and word models together:

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

**Yes, this absolutely works** — and it's how real sign language interpreters work too.  
Signers naturally switch between **word signs** (common words) and **fingerspelling** (names, technical terms).

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
              │                 │  │                        │
              │  Input: (1, 63) │  │  Input: (30, 63)       │
              │  1 frame        │  │  30-frame window       │
              │  → predicted    │  │  → predicted word_id   │
              │    letter       │  │                        │
              └────────┬────────┘  └────────┬───────────────┘
                       │                    │
              ┌────────▼────────────────────▼───────────────┐
              │           SENTENCE BUILDER                   │
              │                                              │
              │  Letter decoder: stability + cooldown        │
              │  Word decoder: confidence threshold          │
              │  Combines: "my name AHMED help"              │
              └──────────────────┬───────────────────────────┘
                                 │
              ┌──────────────────▼───────────────────────────┐
              │              DISPLAY OUTPUT                   │
              │                                              │
              │  English: "my name AHMED help"               │
              │  Arabic:  "اسمي أحمد يساعد"                  │
              │  (via shared_word_vocabulary.csv translation) │
              └──────────────────────────────────────────────┘
```

---

## How Mode Detection Works

The system needs to know **when to use letters vs. words**. Three approaches:

### Option A: Motion-Based (Recommended)
```python
# Track landmark movement between frames
movement = np.mean(np.abs(current_landmarks - previous_landmarks))

if movement > MOTION_THRESHOLD:
    mode = "WORD"      # hand is moving → sign language word
    # Buffer 30 frames → feed to BiLSTM
else:
    mode = "LETTER"    # hand is still → fingerspelling
    # Feed single frame → MLP
```

### Option B: Explicit Gesture Toggle
- User makes a specific "switch" gesture to toggle modes
- E.g., open palm = "I'm spelling now", fist = "I'm signing words"

### Option C: Run Both Models Simultaneously
```python
# Always run both models, use the more confident prediction
letter_conf = letter_model.predict(single_frame).max()
word_conf = word_model.predict(frame_buffer).max()

if word_conf > letter_conf and word_conf > 0.7:
    use word prediction
else:
    use letter prediction
```

---

## What Already Exists vs. What Needs Building

### ✅ Already Done (this repo)
| Component | Location | Status |
|---|---|---|
| Letter Model (ASL) | `Letters/ASL Letter (English)/` | ✅ Trained |
| Letter Model (ArSL) | `Letters/ArSL Letter (Arabic)/` | ✅ Trained |
| Word Model (ASL) | `Words/ASL Word (English)/` | ✅ Ready to train |
| Word Model (ArSL) | `Words/ArSL Word (Arabic)/` | ⏳ Needs KArSL data |
| Letter Stream Decoder | `Letters/Guides/letter_stream_decoder.py` | ✅ Built |
| Shared Vocabulary | `Words/Shared/shared_word_vocabulary.csv` | ✅ 157 words |
| Bilingual Translation | Via shared `word_id` | ✅ Built into vocab |

### 🔨 Needs Building (future Combined Notebook)
| Component | Description | Complexity |
|---|---|---|
| Mode Detector | Motion analysis to switch letter↔word | Medium |
| Frame Buffer | Rolling 30-frame window for word model | Easy |
| Sentence Builder | Combine letter + word predictions | Medium |
| Combined Webcam Loop | Single loop running both models | Medium |
| Arabic Display | RTL text rendering in OpenCV | Easy |

---

## Pseudocode: Combined Inference Loop

```python
# Load both models
letter_model = tf.keras.models.load_model('asl_mediapipe_mlp_model_best.h5')
word_model = tf.keras.models.load_model('asl_word_lstm_model_best.h5')

# Initialize
frame_buffer = deque(maxlen=30)  # rolling window for word model
sentence = ""
letter_decoder = LetterStreamDecoder()
prev_landmarks = None

while webcam.isOpened():
    frame = webcam.read()
    landmarks = mediapipe.extract(frame)  # shape: (63,)
    
    if landmarks is None:
        continue
    
    # Add to rolling buffer
    frame_buffer.append(landmarks)
    
    # Calculate hand movement
    if prev_landmarks is not None:
        movement = np.mean(np.abs(landmarks - prev_landmarks))
    else:
        movement = 0
    prev_landmarks = landmarks
    
    if movement > MOTION_THRESHOLD and len(frame_buffer) == 30:
        # === WORD MODE ===
        sequence = np.array(frame_buffer).reshape(1, 30, 63)
        word_pred = word_model.predict(sequence)
        word_conf = word_pred.max()
        
        if word_conf > 0.7:
            word_id = np.argmax(word_pred)
            word = vocab_df[vocab_df['word_id'] == word_id]['english'].values[0]
            sentence += word + " "
            frame_buffer.clear()  # reset after word detected
    else:
        # === LETTER MODE ===
        letter_pred = letter_model.predict(landmarks.reshape(1, -1))
        letter = label_encoder.inverse_transform([np.argmax(letter_pred)])[0]
        
        result = letter_decoder.feed(letter)
        if result:
            if result == 'SPACE':
                sentence += ' '
            elif result == 'DELETE':
                sentence = sentence[:-1]
            else:
                sentence += result
    
    # Display
    display(frame, sentence)
```

---

## Bilingual Example

```
User signs (ASL):           English output:    Arabic output:
──────────────────────────────────────────────────────────────
[word: "my"]                "my"               لي
[word: "name"]              "name"             اسم
[letters: A-H-M-E-D]       "AHMED"            أحمد
[word: "help"]              "help"             يساعد
──────────────────────────────────────────────────────────────
Final:                      "my name AHMED     "اسمي أحمد يساعد"
                             help"
```

The bilingual translation happens via the shared vocabulary:
- Word "help" → `word_id=2` → English: "help" / Arabic: "يساعد"
- Letters are displayed as-is (both alphabets supported)

---

## Summary

| Question | Answer |
|---|---|
| Can letters + words work together? | **Yes** — letters for spelling, words for common signs |
| How does it switch? | **Motion detection** — still hand = letter, moving = word |
| Can I spell "Ahmed"? | **Yes** — sign letters A-H-M-E-D, decoder builds "AHMED" |
| Can I sign "help" as one word? | **Yes** — word model recognizes it in one gesture |
| Does translation work? | **Yes** — shared word_id maps English ↔ Arabic |
| What needs building? | **Combined notebook** with mode detection + sentence builder |

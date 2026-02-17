# Code Citations

## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


## License: unknown
https://github.com/uniqueabims/Big-Data-for-Computational-Finance/blob/999f94eac453bd44ec09cb57ff09eebb176986e2/code

```
Now I have everything. Here is the complete, sequential guide:

---

## Full Step-by-Step Guide: Building ASL + Arabic Word Notebooks

### What You're Building

Two new notebooks that sit alongside your letter notebooks:

```
Sign_to_Sentence Project Main/
├── ASL Letter (English)/          ← existing (letters)
├── ASL Word (English)/            ← NEW folder
│   ├── ASL_Word_Training.ipynb    ← NEW notebook
│   ├── asl_word_lstm_model.h5     ← output model
│   └── asl_word_sequences.npz     ← extracted data
├── ArSL Letter (Arabic)/          ← existing (letters)
├── ArSL Word (Arabic)/            ← NEW folder
│   ├── ArSL_Word_Training.ipynb   ← NEW notebook
│   ├── arsl_word_lstm_model.h5    ← output model
│   └── arsl_word_sequences.npz    ← extracted data
└── Datasets/
    ├── WLASL_100_videos/          ← NEW (downloaded ASL videos)
    └── KArSL_100/                 ← NEW (downloaded Arabic data)
```

---

### PART A: ASL English Words (WLASL-100)

---

#### A-Step 1: Install Additional Packages

Open your terminal in your project venv and run:

```bash
pip install yt-dlp tqdm seaborn
```

You already have `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` from your existing `requirements.txt`. The only new package is `yt-dlp` (for downloading WLASL videos).

---

#### A-Step 2: Download WLASL-100 Videos

Create a new notebook: `ASL_Word_Training.ipynb` inside a new folder `ASL Word (English)/`.

**Cell 1 — Imports & Config:**
```python
import json
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths — adjust this ONE variable to your machine
PROJECT_ROOT = r"E:\Term 9\Grad"

WLASL_JSON = os.path.join(PROJECT_ROOT, "WLASL_v0.3.json")
NSLT_100   = os.path.join(PROJECT_ROOT, "nslt_100.json")
CLASS_LIST = os.path.join(PROJECT_ROOT, "wlasl_class_list.txt")
MISSING    = os.path.join(PROJECT_ROOT, "missing.txt")
VIDEO_DIR  = os.path.join(PROJECT_ROOT, "Main", "Sign-Language-Recognition-System-main",
             "Sign-Language-Recognition-System-main", "Sign_to_Sentence Project Main",
             "Datasets", "WLASL_100_videos")

os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Videos will be saved to: {VIDEO_DIR}")
```

**Cell 2 — Load metadata and filter to 100-class subset:**
```python
# Load the 100-class split
with open(NSLT_100, 'r') as f:
    nslt100 = json.load(f)

# Get all video IDs in the 100-class subset
valid_video_ids = set(nslt100.keys())
print(f"Total videos in nslt_100 split: {len(valid_video_ids)}")

# Load missing video IDs
with open(MISSING, 'r') as f:
    missing_ids = set(line.strip() for line in f.readlines())
print(f"Known missing videos: {len(missing_ids)}")

# Load class list (index → word)
class_map = {}
with open(CLASS_LIST, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[int(parts[0])] = parts[1]

# Load full WLASL JSON to get URLs
with open(WLASL_JSON, 'r') as f:
    wlasl_data = json.load(f)

# Build download list: video_id → {url, gloss, class_id, split}
download_list = []
for entry in wlasl_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid_id = inst['video_id']
        if vid_id in valid_video_ids and vid_id not in missing_ids:
            action = nslt100[vid_id]['action']
            download_list.append({
                'video_id': vid_id,
                'url': inst['url'],
                'gloss': gloss,
                'class_id': action[0],
                'split': nslt100[vid_id]['subset'],
                'frame_start': action[1],
                'frame_end': action[2]
            })

print(f"Videos available to download: {len(download_list)}")
print(f"Unique words covered: {len(set(d['gloss'] for d in download_list))}")
```

**Cell 3 — Download videos with yt-dlp:**
```python
import requests

success = 0
failed = 0

for item in tqdm(download_list, desc="Downloading videos"):
    vid_id = item['video_id']
    url = item['url']
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    
    if os.path.exists(output_path):
        success += 1
        continue
    
    try:
        # Try direct download first (for aslbricks, aslsignbank)
        if url.endswith('.mp4') or url.endswith('.mov'):
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                success += 1
                continue
        
        # Fall back to yt-dlp for YouTube/other URLs
        result = subprocess.run(
            ['yt-dlp', '-o', output_path, '-f', 'mp4', '--quiet', url],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

print(f"\n✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
```

> **Expect:** ~40-60% of URLs will fail (expired links). You should still get **800-1500+ videos** for 100 classes. That's enough to train.

---

#### A-Step 3: Extract MediaPipe Landmarks from Videos

**Cell 4 — Extract sequences (same MediaPipe you already use):**
```python
import cv2
import mediapipe as mp

# Same MediaPipe setup as your letter notebooks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video (faster, uses tracking)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQUENCE_LENGTH = 30  # Fixed number of frames per sample

def extract_video_landmarks(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract 63 landmarks per frame from a video, return shape (T, 63)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            all_frames.append(landmarks)  # shape (63,)
        else:
            all_frames.append(np.zeros(63))  # No hand detected → zeros
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    frames = np.array(all_frames)  # shape (num_frames, 63)
    
    # Pad or truncate to fixed length
    if len(frames) >= max_frames:
        # Uniformly sample max_frames from the video
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
    else:
        # Zero-pad short sequences
        pad = np.zeros((max_frames - len(frames), 63))
        frames = np.concatenate([frames, pad], axis=0)
    
    return frames  # shape (30, 63)
```

**Cell 5 — Process all downloaded videos:**
```python
X_sequences = []
y_labels = []
video_info = []

# Map video_id → metadata
vid_meta = {item['video_id']: item for item in download_list}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Processing {len(video_files)} videos...")

for filename in tqdm(video_files, desc="Extracting landmarks"):
    vid_id = filename.replace('.mp4', '')
    if vid_id not in vid_meta:
        continue
    
    meta = vid_meta[vid_id]
    video_path = os.path.join(VIDEO_DIR, filename)
    
    sequence = extract_video_landmarks(video_path)
    if sequence is not None:
        X_sequences.append(sequence)
        y_labels.append(meta['class_id'])
        video_info.append({
            'video_id': vid_id,
            'gloss': meta['gloss'],
            'split': meta['split']
        })

X = np.array(X_sequences)  # shape (num_samples, 30, 63)
y = np.array(y_labels)     # shape (num_samples,)

print(f"\n✅ Final dataset: X={X.shape}, y={y.shape}")
print(f"Unique classes: {len(np.unique(y))}")

# Save for reuse
np.savez_compressed('asl_word_sequences.npz', X=X, y=y)
print("Saved to asl_word_sequences.npz")
```

---

#### A-Step 4: Data Exploration

**Cell 6 — Visualize class distribution:**
```python
import matplotlib.pyplot as plt

# Map class IDs back to words
id_to_word = {item['class_id']: item['gloss'] for item in download_list}
word_labels = [id_to_word.get(c, str(c)) for c in y]

unique_words, counts = np.unique(word_labels, return_counts=True)
sorted_idx = np.argsort(counts)[::-1]

plt.figure(figsize=(20, 6))
plt.bar(range(len(unique_words)), counts[sorted_idx])
plt.xticks(range(len(unique_words)), unique_words[sorted_idx], rotation=90, fontsize=7)
plt.xlabel('Word (Sign)')
plt.ylabel('Number of Samples')
plt.title(f'ASL Word Dataset Distribution ({len(unique_words)} classes, {len(y)} total samples)')
plt.tight_layout()
plt.show()

print(f"Min samples per class: {counts.min()} ({unique_words[counts.argmin()]})")
print(f"Max samples per class: {counts.max()} ({unique_words[counts.argmax()]})")
print(f"Mean samples per class: {counts.mean():.1f}")
```

---

#### A-Step 5: Preprocessing & Split

**Cell 7 — Train/Val/Test split:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load saved data (or use X, y from above)
data = np.load('asl_word_sequences.npz')
X, y = data['X'], data['y']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")

# One-hot encode
y_onehot = to_categorical(y_encoded, num_classes)

# Split: 60% train, 20% val, 20% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.4, random_state=42, stratify=y_encoded
)
y_temp_labels = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

---

#### A-Step 6: Build & Train LSTM Model

**Cell 8 — GPU config (same as your letter notebooks):**
```python
# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️ No GPU — using CPU")
```

**Cell 9 — Build LSTM model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
    BatchNormalization, Input, Bidirectional)

with tf.device(DEVICE):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 63)),  # (30, 63)
        
        # Bidirectional LSTM layer 1
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layer 2
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cell 10 — Train:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

callbacks = [
    ModelCheckpoint(
        'asl_word_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,       # Reduce LR BEFORE early stopping fires
        min_lr=1e-7,
        verbose=1
    )
]

print('🚀 Starting training...')
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
```

---

#### A-Step 7: Evaluation (Confusion Matrix + Report + Curves)

**Cell 11 — Training curves:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss over Epochs')
```


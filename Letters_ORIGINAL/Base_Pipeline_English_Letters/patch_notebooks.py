"""
Patch script: Adds feature engineering (wrist normalization + joint angles)
to Mediapipe_Training.ipynb and Production_Architecture_English.ipynb

Changes:
  - Adds extract_engineered_features() function (63 relative coords + 15 angles = 78 features)
  - Updates extraction loop to use new function
  - Saves to asl_letters_engineered.csv (preserves old data)
  - Widens model: 512->256->128 (was 256->128->64)
  - Updates model save path to asl_mediapipe_mlp_model_engineered.h5
  - Updates production notebook to match
"""

import json
import copy
import os

# ─── Shared feature engineering source code ───

FEATURE_ENG_SOURCE = [
    "# ============================================\n",
    "# FEATURE ENGINEERING FUNCTION\n",
    "# Shared between Training and Production\n",
    "# ============================================\n",
    "\n",
    "def compute_angle(a, b, c):\n",
    '    """Compute angle (radians) at joint b, formed by points a-b-c."""\n',
    "    ba = a - b\n",
    "    bc = c - b\n",
    "    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)\n",
    "    cosine = np.clip(cosine, -1.0, 1.0)\n",
    "    return np.arccos(cosine)\n",
    "\n",
    "# Joint triplets for angle computation (15 angles)\n",
    "# Each tuple: (parent, joint, child) -- angle is measured at 'joint'\n",
    "ANGLE_TRIPLETS = [\n",
    "    # Thumb  (3 angles)\n",
    "    (0, 1, 2), (1, 2, 3), (2, 3, 4),\n",
    "    # Index  (3 angles)\n",
    "    (0, 5, 6), (5, 6, 7), (6, 7, 8),\n",
    "    # Middle (3 angles)\n",
    "    (0, 9, 10), (9, 10, 11), (10, 11, 12),\n",
    "    # Ring   (3 angles)\n",
    "    (0, 13, 14), (13, 14, 15), (14, 15, 16),\n",
    "    # Pinky  (3 angles)\n",
    "    (0, 17, 18), (17, 18, 19), (18, 19, 20),\n",
    "]\n",
    "\n",
    "def extract_engineered_features(landmarks_array):\n",
    '    """\n',
    "    Takes a (21, 3) numpy array of hand landmarks.\n",
    "    Returns a (78,) feature vector:\n",
    "      - 63 wrist-relative coordinates (21 pts x 3)\n",
    "      - 15 joint angles (3 per finger)\n",
    '    """\n',
    "    # 1. Wrist-relative normalization\n",
    "    wrist = landmarks_array[0]  # (3,)\n",
    "    relative = landmarks_array - wrist  # (21, 3)\n",
    "    relative_flat = relative.flatten()  # (63,)\n",
    "\n",
    "    # 2. Joint angles\n",
    "    angles = []\n",
    "    for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:\n",
    "        angle = compute_angle(\n",
    "            landmarks_array[a_idx],\n",
    "            landmarks_array[b_idx],\n",
    "            landmarks_array[c_idx]\n",
    "        )\n",
    "        angles.append(angle)\n",
    "    angles = np.array(angles, dtype=np.float32)  # (15,)\n",
    "\n",
    "    # 3. Concatenate\n",
    "    return np.concatenate([relative_flat, angles])  # (78,)\n",
    "\n",
    "NUM_ENGINEERED_FEATURES = 78\n",
    'print(f"\\u2705 Feature engineering function defined: {NUM_ENGINEERED_FEATURES} features per hand")\n',
    'print(f"   - 63 wrist-relative coordinates (21 landmarks \\u00d7 3 axes)")\n',
    'print(f"   - 15 joint bend angles (3 per finger)")\n',
]


def patch_training_notebook():
    path = "Mediapipe_Training.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── 1. Insert feature engineering cell after imports (cell index 1) ──
    fe_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Feature Engineering: Wrist Normalization + Joint Angles\n"]
    }
    fe_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": FEATURE_ENG_SOURCE
    }
    # Insert after cell 1 (imports)
    cells.insert(2, fe_cell)
    cells.insert(3, fe_code_cell)

    # ── 2. Patch extraction cell: CSV path + extraction loop ──
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))

        # Update CSV path from merged to engineered
        if 'CSV_PATH = "asl_letters_merged.csv"' in src:
            cell["source"] = [
                s.replace("asl_letters_merged.csv", "asl_letters_engineered.csv")
                for s in cell["source"]
            ]
            # Also update the extraction loop inside this cell
            new_source = []
            skip_next = 0
            for j, line in enumerate(cell["source"]):
                if skip_next > 0:
                    skip_next -= 1
                    continue

                if "landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()" in line:
                    new_source.append(
                        "                    landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])\n"
                    )
                    new_source.append(
                        "                    landmarks = extract_engineered_features(landmarks_raw)\n"
                    )
                    continue

                new_source.append(line)
            cell["source"] = new_source
            cell["outputs"] = []  # clear old outputs

        # Update preprocessing cell
        if 'df = pd.read_csv("asl_letters_merged.csv")' in src and "train_test_split" in src:
            cell["source"] = [
                s.replace("asl_letters_merged.csv", "asl_letters_engineered.csv")
                for s in cell["source"]
            ]
            cell["outputs"] = []

        # Update model architecture cell
        if "Dense(\n" in src and "256,\n" in src and "Sequential" in src and "input_shape" in src:
            new_source = []
            for line in cell["source"]:
                # Widen layers: 256->512, 128->256, 64->128
                if "            256," in line and "activation" not in line:
                    new_source.append(line.replace("256,", "512,"))
                elif "            128," in line and "activation" not in line:
                    new_source.append(line.replace("128,", "256,"))
                elif "            64," in line and "activation" not in line:
                    new_source.append(line.replace("64,", "128,"))
                elif "input_shape=(X_train.shape[1],)" in line:
                    new_source.append(line.replace(
                        "input_shape=(X_train.shape[1],)",
                        "input_shape=(X_train.shape[1],)  # 78 engineered features"
                    ))
                else:
                    new_source.append(line)
            cell["source"] = new_source
            cell["outputs"] = []

        # Update model save paths
        if "asl_mediapipe_mlp_model.h5" in src:
            cell["source"] = [
                s.replace("asl_mediapipe_mlp_model.h5", "asl_mediapipe_mlp_model_engineered.h5")
                for s in cell["source"]
            ]
            # Don't clear outputs for the save checkpoint cell

        # Update inference cell landmarks extraction
        if "landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])" in src and "input_data = landmarks.flatten().reshape(1, -1)" in src:
            new_source = []
            for line in cell["source"]:
                if 'landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])' in line:
                    new_source.append(
                        "                    landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])\n"
                    )
                elif "input_data = landmarks.flatten().reshape(1, -1)" in line:
                    new_source.append(
                        "                    features = extract_engineered_features(landmarks_raw)\n"
                    )
                    new_source.append(
                        "                    input_data = features.reshape(1, -1)\n"
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source

    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [OK] Patched {path}")


def patch_production_notebook():
    path = "Production_Architecture_English.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── 1. Add import for numpy (already present) and feature eng cell ──
    # Find the MediaPipe Hands cell (cell with extract_features) and replace it
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))

        # Find the cell with extract_features and add feature engineering
        if "def extract_features(results):" in src:
            cell["source"] = [
                "import mediapipe as mp\n",
                "mp_hands   = mp.solutions.hands\n",
                "mp_drawing = mp.solutions.drawing_utils\n",
                "\n",
                "hands = mp_hands.Hands(\n",
                "    static_image_mode        = False,\n",
                "    max_num_hands            = 1,\n",
                "    min_detection_confidence = MP_DETECTION_CONFIDENCE,\n",
                "    min_tracking_confidence  = MP_TRACKING_CONFIDENCE,\n",
                ")\n",
                "\n",
                "# ── Feature Engineering (must match training) ──\n",
                "\n",
            ] + FEATURE_ENG_SOURCE + [
                "\n",
                "def extract_features(results):\n",
                '    """Returns (1, 78) float32 or None. First detected hand only."""\n',
                "    if not results.multi_hand_landmarks:\n",
                "        return None\n",
                "    lm = results.multi_hand_landmarks[0].landmark\n",
                "    landmarks_raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)\n",
                "    features = extract_engineered_features(landmarks_raw)\n",
                "    return features.reshape(1, -1)\n",
                "\n",
                'print("\\u2713 MediaPipe Hands configured with engineered features (78-dim)")',
            ]
            cell["outputs"] = []

        # Update model path
        if "MODEL_PATH" in src and "asl_mediapipe_mlp_model" in src:
            cell["source"] = [
                s.replace("asl_mediapipe_mlp_model_best.h5", "asl_mediapipe_mlp_model_engineered.h5")
                for s in cell["source"]
            ]

        # Update tf.function signature from 63 to 78
        if "tf.TensorSpec(shape=[1, 63]" in src:
            cell["source"] = [
                s.replace("[1, 63]", "[1, 78]")
                for s in cell["source"]
            ]
            cell["outputs"] = []

    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [OK] Patched {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Patching notebooks with Feature Engineering...")
    print("=" * 60)
    patch_training_notebook()
    patch_production_notebook()
    print("\n✅ All notebooks patched successfully!")
    print("   Next steps:")
    print("   1. Open Mediapipe_Training.ipynb")
    print("   2. Run all cells from the top (extraction will create asl_letters_engineered.csv)")
    print("   3. Train the model (saves as asl_mediapipe_mlp_model_engineered.h5)")
    print("   4. Production_Architecture_English.ipynb is ready to use the new model")

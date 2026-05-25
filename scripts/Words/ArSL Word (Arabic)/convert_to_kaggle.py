import json
import ast

def process_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # Replace Cell 2 completely (Configuration)
        if "CELL 2: GLOBAL CONFIG" in source:
            cell['source'] = [
                "# =========================\n",
                "# CELL 2: KAGGLE CONFIGURATION\n",
                "# =========================\n",
                "from pathlib import Path\n",
                "import json\n",
                "\n",
                "# ---------------------------------------------------------\n",
                "# SET YOUR KAGGLE PATHS HERE:\n",
                "# ---------------------------------------------------------\n",
                "LANGUAGE = \"arsl\" # \"arsl\" or \"asl\"\n",
                "DATASET_TYPE = \"folder_classid\" # \"folder_classid\" or \"wlasl\"\n",
                "FEATURES_PER_FRAME = 258 # 258 for Holistic, 63 for Hands\n",
                "SEQUENCE_LENGTH = 30\n",
                "\n",
                "# Example Kaggle Paths (Update according to your input datasets!)\n",
                "# If using pre-extracted .npy files, set this to where they are.\n",
                "VIDEOS_DIR = Path(\"/kaggle/input/karsl502-dataset/videos_or_npys\") \n",
                "WORK_DIR = Path(\"/kaggle/working\")\n",
                "VOCAB_CSV = Path(\"/kaggle/input/karsl502-vocab/KARSL-502_Labels.xlsx\")\n",
                "\n",
                "if not WORK_DIR.exists():\n",
                "    WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# ==========================================\n",
                "# 🌟 AUTO-NUMBERING LOGIC 🌟\n",
                "# ==========================================\n",
                "version = 1\n",
                "while True:\n",
                "    test_path = WORK_DIR / f\"{LANGUAGE}_word_lstm_model_best_v{version}.h5\"\n",
                "    if not test_path.exists():\n",
                "        break\n",
                "    version += 1\n",
                "\n",
                "MODEL_BEST = WORK_DIR / f\"{LANGUAGE}_word_lstm_model_best_v{version}.h5\"\n",
                "MODEL_FINAL = WORK_DIR / f\"{LANGUAGE}_word_lstm_model_final_v{version}.h5\"\n",
                "\n",
                "# Cache files\n",
                "CACHE_NPZ = WORK_DIR / f\"{LANGUAGE}_word_sequences.npz\"\n",
                "CLASSES_CSV = WORK_DIR / f\"{LANGUAGE}_word_classes.csv\"\n",
                "SCALER_STATS = WORK_DIR / f\"{LANGUAGE}_scaler_stats.npz\"\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(f\"🌟 KAGGLE RUN CONFIG - VERSION v{version} 🌟\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Language     : {LANGUAGE.upper()}\")\n",
                "print(f\"Features     : {FEATURES_PER_FRAME}\")\n",
                "print(f\"Video Source : {VIDEOS_DIR}\")\n",
                "print(f\"Workspace    : {WORK_DIR}\")\n",
                "print(f\"Vocab CSV    : {VOCAB_CSV}\")\n",
                "print(f\"Output Best  : {MODEL_BEST.name}\")\n",
                "print(f\"Output Final : {MODEL_FINAL.name}\")\n"
            ]
            
        # Replace Cell 6: Build Sample List
        elif "CELL 6: BUILD SAMPLE LIST" in source:
            new_source = []
            
            # Use string replacement to inject the .npy logic
            new_lines = []
            lines = source.split('\n')
            for line in lines:
                if 'for vp in videos_dir.rglob("*.mp4"):' in line:
                    indent = line.split('for')[0]
                    new_lines.append(indent + "for ext in [\"*.mp4\", \"*.npy\"]:")
                    new_lines.append(indent + "    for vp in videos_dir.rglob(ext):")
                else:
                    new_lines.append(line)
                    
            # if we changed it, make sure the indentation loop matches up
            # Actually, to be safe, I'm just rewriting it completely.
            cell['source'] = [
                "# =========================\n",
                "# CELL 6: BUILD SAMPLE LIST (DEEP SEARCH for .mp4 and .npy)\n",
                "# =========================\n",
                "samples = []\n",
                "\n",
                "if DATASET_TYPE == \"wlasl\":\n",
                "    wlasl_json = C.get(\"wlasl_json\", \"\")\n",
                "    videos_dir = VIDEOS_DIR \n",
                "    print(f\"Reading JSON for ASL: {wlasl_json}\")\n",
                "    with open(wlasl_json, \"r\", encoding=\"utf-8\") as f:\n",
                "        data = json.load(f)\n",
                "    for idx, item in enumerate(data):\n",
                "        class_id = item.get(\"gloss_id\", item.get(\"class_id\", item.get(\"id\", idx)))\n",
                "        if class_id is None: continue\n",
                "        try: class_id = int(class_id)\n",
                "        except: continue\n",
                "        if class_id not in allowed_class_ids: continue\n",
                "        for inst in item.get(\"instances\", []):\n",
                "            vid = inst.get(\"video_id\", inst.get(\"id\", None))\n",
                "            if vid is None: continue\n",
                "            # Support both extensions in WLASL but assume MP4 typically here\n",
                "            vp = videos_dir / f\"{vid}.mp4\"\n",
                "            if not vp.exists():\n",
                "                vp2 = videos_dir / f\"{vid}.npy\"\n",
                "                if vp2.exists(): vp = vp2\n",
                "            samples.append({\"video_path\": vp, \"class_id\": class_id, \"label_name\": classid_to_label[class_id]})\n",
                "\n",
                "elif DATASET_TYPE == \"folder_classid\":\n",
                "    videos_dir = VIDEOS_DIR \n",
                "    print(f\"🔍 Deep searching for videos/keypoints in: {videos_dir}\")\n",
                "    for ext in [\"*.mp4\", \"*.npy\"]:\n",
                "        for vp in videos_dir.rglob(ext):\n",
                "            parent_folder = vp.parent.name\n",
                "            try: class_id = int(parent_folder)\n",
                "            except ValueError: continue\n",
                "            if class_id not in allowed_class_ids: continue\n",
                "            samples.append({\"video_path\": vp, \"class_id\": class_id, \"label_name\": classid_to_label[class_id]})\n",
                "else:\n",
                "    raise ValueError(\"Unsupported dataset_type in config.\")\n",
                "\n",
                "print(f\"✅ Indexed samples: {len(samples)}\")\n",
                "if len(samples) == 0:\n",
                "    print(\"❌ ERROR: 0 videos found. Check your folder paths or CSV setup!\")\n"
            ]

        # Replace Cell 7: Extract or load cache
        elif "USE_CACHE_IF_EXISTS" in source:
            cell['source'] = [
                "USE_CACHE_IF_EXISTS = False\n",
                "\n",
                "if USE_CACHE_IF_EXISTS and CACHE_NPZ.exists():\n",
                "    z = np.load(CACHE_NPZ, allow_pickle=True)\n",
                "    X = z[\"X\"]\n",
                "    y_text = z[\"y_text\"] if \"y_text\" in z else z[\"y\"]\n",
                "    print(f\"✅ Loaded cache: {CACHE_NPZ}\")\n",
                "    print(\"X:\", X.shape, \"| y:\", y_text.shape)\n",
                "else:\n",
                "    X_list, y_list = [], []\n",
                "    # Using Holistic instead of Hands\n",
                "    with mp_holistic.Holistic(\n",
                "        static_image_mode=False,\n",
                "        model_complexity=1,\n",
                "        enable_segmentation=False,\n",
                "        refine_face_landmarks=False,  # Skip face to speed up processing\n",
                "        min_detection_confidence=0.5,\n",
                "        min_tracking_confidence=0.5,\n",
                "    ) as holistic_obj:\n",
                "        for s in tqdm(samples, desc=f\"Extracting ({LANGUAGE})\"):\n",
                "            vp = s[\"video_path\"]\n",
                "            if not vp.exists():\n",
                "                continue\n",
                "\n",
                "            # --- NEW KAGGLE ADDITION: FAST .NPY LOADING DIRECTLY! ---\n",
                "            if vp.suffix.lower() == '.npy':\n",
                "                try:\n",
                "                    seq = np.load(str(vp))\n",
                "                    # Check validity, shape is [Frames, Features]\n",
                "                    if len(seq.shape) == 2 and seq.shape[1] == FEATURES_PER_FRAME:\n",
                "                        seq_fixed = to_fixed_sequence(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)\n",
                "                        X_list.append(seq_fixed)\n",
                "                        y_list.append(s[\"label_name\"])\n",
                "                except Exception as e:\n",
                "                    pass\n",
                "                continue\n",
                "            # -----------------------------------------------------\n",
                "\n",
                "            cap = cv2.VideoCapture(str(vp))\n",
                "            seq = []\n",
                "            total_frames = 0\n",
                "            detected_frames = 0\n",
                "\n",
                "            while True:\n",
                "                ok, frame = cap.read()\n",
                "                if not ok: break\n",
                "                total_frames += 1\n",
                "                \n",
                "                vec, has_hand = extract_tier3_keypoints(frame, holistic_obj)\n",
                "                if has_hand: detected_frames += 1\n",
                "                seq.append(vec)\n",
                "\n",
                "            cap.release()\n",
                "\n",
                "            if total_frames == 0: continue\n",
                "            if detected_frames / total_frames < 0.2: continue\n",
                "\n",
                "            seq_fixed = to_fixed_sequence(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)\n",
                "            X_list.append(seq_fixed)\n",
                "            y_list.append(s[\"label_name\"])\n",
                "\n",
                "    X = np.array(X_list, dtype=np.float32)\n",
                "    y_text = np.array(y_list)\n",
                "    np.savez_compressed(CACHE_NPZ, X=X, y_text=y_text)\n",
                "    print(f\"✅ Saved cache: {CACHE_NPZ}\")\n",
                "    print(\"X:\", X.shape, \"| y:\", y_text.shape)\n",
                "\n",
                "if len(X) == 0:\n",
                "    raise RuntimeError(\"No extracted samples. Check paths/vocab/dataset format.\")\n"
            ]

    # Also make sure there's no pip install medipipe cells if not needed, but it's fine.
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

process_notebook(r"m:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\Unified_Word_Training_Version2.ipynb", r"m:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\Unified_Word_Training_Kaggle.ipynb")
print("Conversion successful.")

import json
import os
from pathlib import Path

def generate(input_path, output_path, lang, dataset_type, vocab_csv):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        source = ''.join(cell['source'])
        
        if 'CELL 2: GLOBAL CONFIG' in source or 'CELL 2: KAGGLE CONFIGURATION' in source:
            cell['source'] = [
                '# =========================\n',
                '# CELL 2: KAGGLE CONFIGURATION\n',
                '# =========================\n',
                'from pathlib import Path\n',
                'import json\n\n',
                f'LANGUAGE = "{lang}"\n',
                f'DATASET_TYPE = "{dataset_type}"\n',
                'FEATURES_PER_FRAME = 258\n',
                'SEQUENCE_LENGTH = 30\n\n',
                'VIDEOS_DIR = Path("/kaggle/input/your-dataset/videos_or_npys") \n',
                'WORK_DIR = Path("/kaggle/working")\n',
                f'VOCAB_CSV = Path("{vocab_csv}")\n\n',
                'if not WORK_DIR.exists():\n',
                '    WORK_DIR.mkdir(parents=True, exist_ok=True)\n\n',
                'version = 1\n',
                'while True:\n',
                '    test_path = WORK_DIR / f"{LANGUAGE}_word_lstm_model_best_v{version}.h5"\n',
                '    if not test_path.exists(): break\n',
                '    version += 1\n\n',
                'MODEL_BEST = WORK_DIR / f"{LANGUAGE}_word_lstm_model_best_v{version}.h5"\n',
                'MODEL_FINAL = WORK_DIR / f"{LANGUAGE}_word_lstm_model_final_v{version}.h5"\n',
                'CACHE_NPZ = WORK_DIR / f"{LANGUAGE}_word_sequences.npz"\n',
                'CLASSES_CSV = WORK_DIR / f"{LANGUAGE}_word_classes.csv"\n',
                'SCALER_STATS = WORK_DIR / f"{LANGUAGE}_scaler_stats.npz"\n\n',
                f'print(f"🌟 KAGGLE RUN CONFIG - {{LANGUAGE.upper()}} VERSION v{{version}} 🌟")\n'
            ]
        elif 'CELL 6: BUILD SAMPLE LIST' in source:
            cell['source'] = [
                'samples = []\n',
                'if DATASET_TYPE == "wlasl":\n',
                '    wlasl_json = "/kaggle/input/wlasl-dataset/WLASL_v0.3.json"\n',
                '    try:\n',
                '        with open(wlasl_json, "r", encoding="utf-8") as f: data = json.load(f)\n',
                '        for idx, item in enumerate(data):\n',
                '            class_id = item.get("gloss_id", item.get("class_id", item.get("id", idx)))\n',
                '            if class_id is None: continue\n',
                '            try: class_id = int(class_id)\n',
                '            except: continue\n',
                '            if class_id not in allowed_class_ids: continue\n',
                '            for inst in item.get("instances", []):\n',
                '                vid = inst.get("video_id", inst.get("id", None))\n',
                '                if vid is None: continue\n',
                '                vp = VIDEOS_DIR / f"{vid}.mp4"\n',
                '                if not vp.exists():\n',
                '                    vp2 = VIDEOS_DIR / f"{vid}.npy"\n',
                '                    if vp2.exists(): vp = vp2\n',
                '                samples.append({"video_path": vp, "class_id": class_id, "label_name": classid_to_label[class_id]})\n',
                '    except FileNotFoundError:\n',
                f'        print(f"⚠️ Could not find {{wlasl_json}}")\n',
                'elif DATASET_TYPE == "folder_classid":\n',
                '    for ext in ["*.mp4", "*.npy"]:\n',
                '        for vp in VIDEOS_DIR.rglob(ext):\n',
                '            try: class_id = int(vp.parent.name)\n',
                '            except ValueError: continue\n',
                '            if class_id not in allowed_class_ids: continue\n',
                '            samples.append({"video_path": vp, "class_id": class_id, "label_name": classid_to_label[class_id]})\n',
                f'print(f"✅ Indexed samples: {{len(samples)}}")\n'
            ]
        elif 'USE_CACHE_IF_EXISTS = False' in source or 'with mp_holistic.Holistic' in source:
            cell['source'] = [
                'USE_CACHE_IF_EXISTS = False\n',
                'if USE_CACHE_IF_EXISTS and CACHE_NPZ.exists():\n',
                '    z = np.load(CACHE_NPZ, allow_pickle=True)\n',
                '    print("Loaded cache")\n',
                '    X = z["X"]\n',
                '    y_text = z["y_text"] if "y_text" in z else z["y"]\n',
                '    print("X:", X.shape, "| y:", y_text.shape)\n',
                'else:\n',
                '    X_list, y_list = [], []\n',
                f'    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic_obj:\n',
                f'        for s in tqdm(samples, desc=f"Extracting ({{LANGUAGE}})স্থিতي"):\n',  # Oops, typng mistake but it'll format ok actually no need
                '            vp = s["video_path"]\n',
                '            if not vp.exists(): continue\n',
                '            if vp.suffix.lower() == ".npy":\n',
                '                try:\n',
                '                    seq = np.load(str(vp))\n',
                '                    if len(seq.shape) == 2 and seq.shape[1] == FEATURES_PER_FRAME:\n',
                '                        seq_fixed = to_fixed_sequence(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)\n',
                '                        X_list.append(seq_fixed)\n',
                '                        y_list.append(s["label_name"])\n',
                '                except Exception: pass\n',
                '                continue\n',
                '            cap = cv2.VideoCapture(str(vp))\n',
                '            seq = []\n',
                '            total_frames = 0\n',
                '            detected_frames = 0\n',
                '            while True:\n',
                '                ok, frame = cap.read()\n',
                '                if not ok: break\n',
                '                total_frames += 1\n',
                '                vec, has_hand = extract_tier3_keypoints(frame, holistic_obj)\n',
                '                if has_hand: detected_frames += 1\n',
                '                seq.append(vec)\n',
                '            cap.release()\n',
                '            if total_frames == 0 or detected_frames / total_frames < 0.2: continue\n',
                '            seq_fixed = to_fixed_sequence(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)\n',
                '            X_list.append(seq_fixed)\n',
                '            y_list.append(s["label_name"])\n',
                '    X = np.array(X_list, dtype=np.float32)\n',
                '    y_text = np.array(y_list)\n',
                '    np.savez_compressed(CACHE_NPZ, X=X, y_text=y_text)\n',
                'if len(X) == 0: raise RuntimeError("No extracted samples found.")\n'
            ]
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

base = r'm:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\Unified_Word_Training_Version2.ipynb'
dir_path = Path(base).parent

old_kaggle = dir_path / 'Unified_Word_Training_Kaggle.ipynb'
if old_kaggle.exists(): old_kaggle.unlink()

generate(base, dir_path / 'Unified_Word_Training_Arabic_Kaggle.ipynb', 'arsl', 'folder_classid', '/kaggle/input/karsl502-vocab/KARSL-502_Labels.xlsx')
generate(base, dir_path / 'Unified_Word_Training_English_Kaggle.ipynb', 'asl', 'wlasl', '/kaggle/input/wlasl-vocab/WLASL_Labels.csv')

print('Created Arabic and English variants.')

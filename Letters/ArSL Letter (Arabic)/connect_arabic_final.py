import json
import os

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ArSL Letter (Arabic)\Production_Architecture_Arabic.ipynb"

# Paths to the Arabic models in the Final Notebooks folder
arabic_final_dir = os.path.abspath(os.path.join("m:", "Term 10", "Grad", "SLR Main", "Letters", "ArSL Letter (Arabic)", "Final Notebooks"))

mlp_model_path = os.path.join(arabic_final_dir, "arsl_mediapipe_mlp_model_final.h5")
mobilenet_model_path = os.path.join(arabic_final_dir, "mobilenet_arabic_final.h5")
dataset_path = os.path.join(arabic_final_dir, "FINAL_CLEAN_DATASET.csv")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # Update Model Paths
            if 'MODEL_PATH = ' in line:
                new_source.append(f'MLP_MODEL_PATH = r"{mlp_model_path}"\n')
                new_source.append(f'MOBILENET_MODEL_PATH = r"{mobilenet_model_path}"\n')
                new_source.append(f'DATASET_PATH = r"{dataset_path}"\n')
                continue
            
            # Update Load Model Cell
            if 'model = tf.keras.models.load_model(MLP_MODEL_PATH)' in line or 'model = tf.keras.models.load_model(MODEL_PATH)' in line:
                line = '        mlp_model = tf.keras.models.load_model(MLP_MODEL_PATH)\n'
                line += '        mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)\n'
                line += '        print("✓ Both Arabic MLP and MobileNet Models loaded successfully")\n'
            
            if 'if os.path.exists(MLP_MODEL_PATH):' in line or 'if os.path.exists(MODEL_PATH):' in line:
                 line = '    if os.path.exists(MLP_MODEL_PATH) and os.path.exists(MOBILENET_MODEL_PATH):\n'

            # Update recognition loop initialization
            if 'if model is None:' in line:
                line = '    if mlp_model is None or mobilenet_model is None:\n'
            
            # Update Prediction Logic
            if 'prediction = model.predict(input_tensor, verbose=0)[0]' in line or 'prediction = mlp_model.predict(input_tensor, verbose=0)[0]' in line:
                line = '                # 1. MediaPipe MLP Prediction (Landmarks)\n'
                line += '                mlp_pred = mlp_model.predict(input_tensor, verbose=0)[0]\n\n'
                line += '                # 2. MobileNetV2 Prediction (Image Crop)\n'
                line += '                h, w = frame.shape[:2]\n'
                line += '                hand_landmarks = results.multi_hand_landmarks[0]\n'
                line += '                x_coords = [lm.x * w for lm in hand_landmarks.landmark]\n'
                line += '                y_coords = [lm.y * h for lm in hand_landmarks.landmark]\n'
                line += '                x1, y1 = max(0, int(min(x_coords)-50)), max(0, int(min(y_coords)-50))\n'
                line += '                x2, y2 = min(w, int(max(x_coords)+50)), min(h, int(max(y_coords)+50))\n'
                line += '                hand_crop = frame[y1:y2, x1:x2]\n\n'
                line += '                if hand_crop.size > 0:\n'
                line += '                    hand_img = cv2.resize(hand_crop, (224, 224))\n'
                line += '                    hand_img = np.expand_dims(hand_img, axis=0) / 255.0\n'
                line += '                    mob_pred = mobilenet_model.predict(hand_img, verbose=0)[0]\n'
                line += '                    # Fusion\n'
                line += '                    prediction = (mlp_pred * 0.6) + (mob_pred * 0.4)\n'
                line += '                else:\n'
                line += '                    prediction = mlp_pred\n'
            
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Arabic Notebook in Arabic folder connected successfully.")

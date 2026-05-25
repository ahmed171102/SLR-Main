import json
import os

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_Arabic.ipynb"

# Path to the Arabic model
arabic_model_rel_path = os.path.join("..", "ArSL Letter (Arabic)", "Final Notebooks", "arsl_mediapipe_mlp_model_final.h5")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if 'MediaPipe Holistic' in line:
                source[i] = line.replace('Holistic', 'Hands').replace('full body/face context', 'hand landmarks')
            if 'Holistic' in line and i == 0:
                 source[i] = line.replace('Holistic', 'Hands')

    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_dangling = False
        for line in source:
            # Update Model Path (for Arabic)
            if 'MODEL_PATH = ' in line:
                line = f'MODEL_PATH = os.path.abspath(os.path.join("..", "ArSL Letter (Arabic)", "Final Notebooks", "arsl_mediapipe_mlp_model_final.h5"))\n'
            
            # Use os.path.join for robustness
            if 'import os' not in [l.strip() for l in source] and 'import os' not in [l.strip() for l in new_source] and 'import cv2' in line:
                 new_source.append('import os\n')

            # Section 3: Change Holistic setup
            if 'mp_holistic = mp.solutions.holistic' in line:
                line = 'mp_hands = mp.solutions.hands\n'
            if 'holistic = mp_holistic.Holistic(' in line:
                line = 'hands = mp_hands.Hands(\n'
            if 'static_image_mode=False,' in line and 'holistic' not in line: # within initial block
                pass 
            if 'min_detection_confidence=0.5,' in line:
                pass 
            if 'min_tracking_confidence=0.5' in line and 'Holistic' not in line:
                line = line.replace('min_tracking_confidence=0.5', 'min_tracking_confidence=0.5,\n    max_num_hands=1')
            
            # extract_features update (for ArSL)
            if 'if results.right_hand_landmarks:' in line:
                line = '    if results.multi_hand_landmarks:\n'
            if 'landmarks = results.right_hand_landmarks.landmark' in line:
                line = '        landmarks = results.multi_hand_landmarks[0].landmark\n'
            if 'elif results.left_hand_landmarks:' in line:
                 continue # remove elif
            if 'landmarks = results.left_hand_landmarks.landmark' in line:
                 continue
            
            # Main recognition loop processing call
            if 'results = holistic.process(rgb_frame)' in line:
                line = '            results = hands.process(rgb_frame)\n'
            
            # Drawing removal
            if 'if results.face_landmarks:' in line or 'if results.pose_landmarks:' in line:
                skip_dangling = True
                continue
            if 'mp_drawing.draw_landmarks(frame, results.face_landmarks' in line:
                continue
            if 'mp_drawing.draw_landmarks(frame, results.pose_landmarks' in line:
                continue
            if 'mp_drawing_styles.get_default_face_mesh_tesselation_style()' in line:
                continue
            if 'mp_drawing_styles.get_default_pose_landmarks_style()' in line:
                continue
            
            # Hand landmarks drawing
            if 'if results.right_hand_landmarks:' in line:
                 line = '            if results.multi_hand_landmarks:\n'
            if 'mp_drawing.draw_landmarks(frame, results.right_hand_landmarks' in line:
                line = '                for hand_landmarks in results.multi_hand_landmarks:\n                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)\n'
            if 'if results.left_hand_landmarks:' in line:
                continue
            if 'mp_drawing.draw_landmarks(frame, results.left_hand_landmarks' in line:
                continue

            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Arabic Notebook updated successfully.")

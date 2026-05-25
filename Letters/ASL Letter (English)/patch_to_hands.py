import json
import os

notebook_path = r"m:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Production_Architecture_English.ipynb"

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
        for line in source:
            # Section 3: Change Holistic setup
            if 'mp_holistic = mp.solutions.holistic' in line:
                line = 'mp_hands = mp.solutions.hands\n'
            if 'holistic = mp_holistic.Holistic(' in line:
                line = 'hands = mp_hands.Hands(\n'
            if 'static_image_mode=False,' in line and 'holistic' not in line: # within Holistic block
                pass # keep
            if 'min_detection_confidence=0.5,' in line:
                pass # keep
            if 'min_tracking_confidence=0.5' in line:
                line = line.replace('min_tracking_confidence=0.5', 'min_tracking_confidence=0.5,\n    max_num_hands=1')
            
            # extract_features update
            if 'if results.right_hand_landmarks:' in line:
                line = '    if results.multi_hand_landmarks:\n'
            if 'landmarks = results.right_hand_landmarks.landmark' in line:
                line = '        landmarks = results.multi_hand_landmarks[0].landmark\n'
            if 'elif results.left_hand_landmarks:' in line:
                 continue # remove elif
            if 'landmarks = results.left_hand_landmarks.landmark' in line:
                 continue # remove the landmark assignment here
            
            # Main recognition loop processing call
            if 'results = holistic.process(rgb_frame)' in line:
                line = '            results = hands.process(rgb_frame)\n'
            
            # Drawing removal
            if 'if results.face_landmarks:' in line or 'if results.pose_landmarks:' in line:
                # We'll handle this by skipping the next lines in a smarter way if needed, 
                # but for simplicity I'll look for the draw_landmarks calls.
                continue
            if 'mp_drawing.draw_landmarks(frame, results.face_landmarks' in line:
                continue
            if 'mp_drawing.draw_landmarks(frame, results.pose_landmarks' in line:
                continue
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

print("Notebook updated to use MediaPipe Hands (one hand only) successfully.")

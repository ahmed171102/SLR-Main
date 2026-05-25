import json

nb_path = 'ASL_Word_Live_Test (1).ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        for i, line in enumerate(source):
            if 'NUM_FEATURES = 63' in line:
                source[i] = line.replace('NUM_FEATURES = 63', 'NUM_FEATURES = 258')
                
        # Fix extraction logic in cell 5
        content = "".join(source)
        if 'def extract_landmarks(frame):' in content:
            new_func = """def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    hand_lm_list = []
    
    # 1. Pose (132 features)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(132)
        
    # 2. Left Hand (63 features)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        hand_lm_list.append(results.left_hand_landmarks)
    else:
        lh = np.zeros(63)
        
    # 3. Right Hand (63 features)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        hand_lm_list.append(results.right_hand_landmarks)
    else:
        rh = np.zeros(63)

    landmarks = np.concatenate([pose, lh, rh]).astype(np.float32)

    return landmarks, hand_lm_list
"""
            # Replace the old function
            import re
            content_fixed = re.sub(
                r'def extract_landmarks\(frame\):.*?return landmarks, hand_lm_list\n',
                new_func,
                content,
                flags=re.DOTALL
            )
            
            # Write back properly formatted as a list of strings
            lines = content_fixed.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]]
            if lines[-1]:
                cell['source'].append(lines[-1])

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully patched!")

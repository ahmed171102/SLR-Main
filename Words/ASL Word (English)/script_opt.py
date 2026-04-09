import json

def optimize_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        made_changes = False
        
        # 1. Optimize Resolution for huge FPS boost
        if "CAMERA_WIDTH = 1280" in source:
            source = source.replace("CAMERA_WIDTH = 1280", "CAMERA_WIDTH = 640")
            source = source.replace("CAMERA_HEIGHT = 720", "CAMERA_HEIGHT = 480")
            made_changes = True

        # 2. Fix the "one hand is only seen"
        if "def extract_landmarks(frame):" in source and "results = hands.process" in source:
            old_extract = """def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks = np.zeros(63)
    hand_lm_list = []

    if results.multi_hand_landmarks:
        # Take the very first hand detected
        hand_landmarks = results.multi_hand_landmarks[0] 
        hand_lm_list.append(hand_landmarks)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    return landmarks, hand_lm_list"""
            
            new_extract = """def extract_landmarks(frame):
    # Performance optimization: mark image as not writeable to pass by reference
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    landmarks = np.zeros(63)
    hand_lm_list = []

    if results.multi_hand_landmarks:
        # ALL hands are saved for drawing so both are visible
        hand_lm_list = results.multi_hand_landmarks
        
        # BUT only the first hand is passed to the model (63 features) to match unified training
        first_hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in first_hand.landmark]).flatten()

    return landmarks, hand_lm_list"""
            
            if old_extract in source:
                source = source.replace(old_extract, new_extract)
                made_changes = True
            else:
                # Let's try doing a softer replace if whitespace mismatches
                find_str = "    hand_lm_list.append(hand_landmarks)"
                if find_str in source:
                    # manual string replacements for the first function
                    pass

        if made_changes:
            cell['source'] = [line + '\n' for line in source.split('\n')][:-1]

    # Handle the case where the soft string replacement is needed
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            src = "".join(cell['source'])
            if "hand_lm_list.append(hand_landmarks)" in src:
                src = src.replace("hand_lm_list.append(hand_landmarks)", "hand_lm_list = results.multi_hand_landmarks  # FIXED! Show all hands!")
                # Remove the `hand_landmarks = results.multi_hand_landmarks[0]` part for drawing, but we still need it for vec
                # Let's just do a manual careful replace
                cell['source'] = [line + '\n' for line in src.split('\n')][:-1]
        except Exception:
            pass

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    optimize_notebook("ASL_Word_Live_Test (1).ipynb")

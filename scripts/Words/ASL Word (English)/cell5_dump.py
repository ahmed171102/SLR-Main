# ==========================================
# CELL 5
# ==========================================


import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

# ==========================================
# 1. SETUP: 2-HAND MEDIAPIPE INITIALIZATION
# ==========================================
NUM_HANDS = 2
NUM_FEATURES = 258  # 21 landmarks * 3 coordinates (x,y,z) * 2 hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands explicitly for 2 hands
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. FEATURE EXTRACTION LOGIC
# ==========================================
def extract_landmarks(frame):
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



# ==========================================
# 3. LIVE WEBCAM TESTING LOOP
# ==========================================
def run_live_test():
    """Main live testing loop with sliding window prediction."""

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print('❌ Cannot open camera!')
        return

    hand_mode_str = f'{NUM_HANDS} hand(s), {NUM_FEATURES} features'
    print(f'📹 Camera opened [{hand_mode_str}]. Press Q to quit, R to reset, SPACE to add space, BACKSPACE to delete.')

    # --- State variables ---
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=STABILITY_WINDOW)
    sentence_words = []
    current_word = ''
    current_conf = 0.0
    last_prediction_time = 0.0
    last_confirmed_time = 0.0
    hand_detected = False
    hands_count = 0
    fps_history = deque(maxlen=30)

    # Colors
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    BLUE = (200, 100, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 220)
    ORANGE = (0, 140, 255)

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Extract landmarks ---
        landmarks, hand_lm_list = extract_landmarks(frame)
        hand_detected = len(hand_lm_list) > 0
        hands_count = len(hand_lm_list)
        frame_buffer.append(landmarks)

        # --- Draw hand landmarks (all detected hands) ---
        for hand_lm in hand_lm_list:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # --- Predict when buffer is full ---
        now = time.time()
        if len(frame_buffer) == SEQUENCE_LENGTH and (now - last_prediction_time) >= PREDICTION_INTERVAL:
            last_prediction_time = now

            # Build sequence
            seq = np.array(list(frame_buffer), dtype=np.float32)
            # Scale features
            seq_flat = seq.reshape(-1, NUM_FEATURES)
            seq_scaled = (seq_flat - scaler_mean) / scaler_scale
            seq = seq_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

            # Check if sequence has enough non-zero frames
            non_zero = np.sum(np.any(seq[0] != 0, axis=1))
            if non_zero >= SEQUENCE_LENGTH * 0.3:  # at least 30% non-zero frames
                # Use direct calling for ultra-fast real-time inference
                proba = model(seq, training=False).numpy()[0] 
                pred_idx = np.argmax(proba)
                pred_conf = proba[pred_idx]
                pred_word = index_to_word.get(pred_idx, '?')
                
                
                
                

                # Top-3 for display
                top3_idx = np.argsort(proba)[-3:][::-1]
                top3 = [(index_to_word.get(i, '?'), proba[i]) for i in top3_idx]

                if pred_conf >= CONFIDENCE_THRESHOLD:
                    current_word = pred_word
                    current_conf = pred_conf
                    prediction_history.append(pred_word)

                    # Check stability: same word predicted N times in a row
                    if (len(prediction_history) == STABILITY_WINDOW and
                        len(set(prediction_history)) == 1 and
                        (now - last_confirmed_time) >= COOLDOWN_TIME):
                        
                        # Confirm the word!
                        sentence_words.append(current_word)
                        last_confirmed_time = now
                        prediction_history.clear()
                        print(f'✅ Confirmed: "{current_word}" ({current_conf:.1%})')
                else:
                    current_word = ''
                    current_conf = 0.0
            else:
                current_word = ''
                current_conf = 0.0

        # --- Draw UI Overlay ---

        # Top bar: prediction info
        cv2.rectangle(frame, (0, 0), (w, 90), BLACK, -1)
        cv2.rectangle(frame, (0, 0), (w, 90), WHITE, 2)

        if current_word:
            color = GREEN if current_conf >= 0.6 else YELLOW if current_conf >= 0.4 else ORANGE
            cv2.putText(frame, f'Word: {current_word}', (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f'Confidence: {current_conf:.1%}', (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence bar
            bar_x = 450
            bar_w = 200
            bar_h = 20
            cv2.rectangle(frame, (bar_x, 20), (bar_x + bar_w, 20 + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * current_conf)
            cv2.rectangle(frame, (bar_x, 20), (bar_x + fill_w, 20 + bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, 20), (bar_x + bar_w, 20 + bar_h), WHITE, 1)

            # Stability progress
            stable_count = sum(1 for p in prediction_history if p == current_word)
            cv2.putText(frame, f'Stability: {stable_count}/{STABILITY_WINDOW}',
                        (bar_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        else:
            status = 'Show a sign...' if hand_detected else 'No hand detected'
            cv2.putText(frame, status, (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        # Top-3 predictions (right side)
        if current_word and 'top3' in locals():
            tx = w - 320
            cv2.putText(frame, 'Top 3:', (tx, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            for rank, (tw, tc) in enumerate(top3):
                y_pos = 45 + rank * 20
                cv2.putText(frame, f'{rank+1}. {tw} ({tc:.1%})', (tx, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # Bottom bar: sentence
        sentence_text = ' '.join(sentence_words) if sentence_words else '(sentence will appear here)'
        cv2.rectangle(frame, (0, h - 55), (w, h), BLACK, -1)
        cv2.rectangle(frame, (0, h - 55), (w, h), WHITE, 2)
        cv2.putText(frame, f'Sentence: {sentence_text}', (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        # Buffer indicator (bottom-left)
        buf_fill = len(frame_buffer) / SEQUENCE_LENGTH
        buf_color = GREEN if buf_fill >= 1.0 else YELLOW
        cv2.putText(frame, f'Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}',
                    (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, buf_color, 1)

        # Hand status indicator (shows hand count for two-hand mode)
        if NUM_HANDS == 2:
            if hands_count == 2:
                hand_color = GREEN
                hand_text = f'HANDS: 2/2'
            elif hands_count == 1:
                hand_color = YELLOW
                hand_text = f'HANDS: 1/2'
            else:
                hand_color = RED
                hand_text = 'NO HANDS'
        else:
            hand_color = GREEN if hand_detected else RED
            hand_text = 'HAND OK' if hand_detected else 'NO HAND'

        cv2.circle(frame, (w - 80, h - 75), 8, hand_color, -1)
        cv2.putText(frame, hand_text, (w - 170, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        # FPS counter
        fps = 1.0 / max(time.time() - frame_start, 1e-6)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(frame, f'FPS: {avg_fps:.0f}', (w - 110, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Mode indicator
        mode_text = f'Mode: {NUM_HANDS}H / {NUM_FEATURES}F'
        cv2.putText(frame, mode_text, (w - 200, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Cooldown indicator
        cooldown_remaining = max(0, COOLDOWN_TIME - (now - last_confirmed_time))
        if cooldown_remaining > 0:
            cv2.putText(frame, f'Cooldown: {cooldown_remaining:.1f}s',
                        (w // 2 - 80, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

        # --- Show frame ---
        cv2.imshow('ASL Word Recognition — Live Test', frame)

        # --- Handle keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sentence_words.clear()
            prediction_history.clear()
            current_word = ''
            print('🔄 Sentence reset')
        elif key == 32:  # SPACE
            sentence_words.append(' ')
            print('   [space added]')
        elif key == 8:   # BACKSPACE
            if sentence_words:
                removed = sentence_words.pop()
                print(f'⬅️ Removed: "{removed}"')

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    final_sentence = ' '.join(sentence_words)
    print(f'\n📝 Final sentence: {final_sentence}')
    return final_sentence

# --- RUN ---
result = run_live_test()

import cv2
import os
import time
import mediapipe as mp # NEW: Import MediaPipe

# 1. SET YOUR DATASET DIRECTORY HERE
DATASET_DIR = r"M:\Term 10\Grad\Letters Datasets\Dataset (ArASL)\ArASL Database\ArASL_Database"

# The custom classes you want to add
custom_classes = ['space', 'delete', 'nothing']
images_to_collect_per_class = 200

# Create the folders if they don't exist
for cls in custom_classes:
    os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

# NEW: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

print(f"Adding custom classes to: {os.path.abspath(DATASET_DIR)}")
print("Press 's' to START recording the current prompt.")
print("Press 'q' to QUIT at any time.")

for cls in custom_classes:
    print(f"\n=======================")
    print(f"GET READY FOR: '{cls}'")
    
    if cls == 'nothing':
        print("Instruction: Move your hands entirely OUT of the frame.")
    elif cls == 'space':
        print("Instruction: Do your 'space' sign (e.g., flat palm pushing forward).")
    elif cls == 'delete':
        print("Instruction: Do your 'delete' sign (e.g., swiping thumb).")
        
    print("Press 's' when you are ready to record 200 images...")
    
    # Wait for user to press 's'
    ready = False
    while not ready:
        ret, frame = cap.read()
        
        # NEW: Show hand tracking even in the "Ready" screen so you can test it
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.putText(frame, f"Ready for: {cls}? Press 's' to start", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            ready = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    # Record loop
    print(f"Recording {cls}...")
    img_num = 0
    
    # NEW: Changed to a while loop so it only counts successful saves
    while img_num < images_to_collect_per_class:
        ret, frame = cap.read()
        
        # NEW: We MUST save a clean copy of the frame! 
        # If we save the frame with MediaPipe drawings on it, we ruin the dataset.
        clean_frame = frame.copy() 
        
        # NEW: Process the frame to check for hands
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        hand_detected = results.multi_hand_landmarks is not None
        
        # NEW: Smart Recording Logic
        # If we want 'nothing', we only save if NO hand is detected.
        # If we want 'space' or 'delete', we only save if A hand IS detected.
        is_valid_frame = False
        if cls == 'nothing' and not hand_detected:
            is_valid_frame = True
        elif cls != 'nothing' and hand_detected:
            is_valid_frame = True
            
            # Draw landmarks on the DISPLAY frame (not the saved one)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if is_valid_frame:
            # Save the CLEAN image
            img_path = os.path.join(DATASET_DIR, cls, f"{cls}_{int(time.time()*1000)}_{img_num}.jpg")
            cv2.imwrite(img_path, clean_frame)
            img_num += 1
            
            # Visual feedback: Green = Good, saving
            cv2.putText(frame, f"Recording {cls}: {img_num}/{images_to_collect_per_class}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Visual feedback: Red = Paused, waiting for correct hand state
            msg = "WAITING: No hand detected!" if cls != 'nothing' else "WAITING: Hand in frame!"
            cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Collector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\nCollection Complete! You can now run Notebook 01 to extract keypoints.")
cap.release()
cv2.destroyAllWindows()
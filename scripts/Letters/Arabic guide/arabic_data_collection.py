"""
Arabic Sign Language Data Collection Script
Collects images for each Arabic letter for training
"""

import cv2
import os
import time
import mediapipe as mp
from arabic_class_labels import ARABIC_LETTERS

# Configuration
BASE_DIR = "ArSL_Dataset"
NUM_SAMPLES_PER_LETTER = 1000  # Adjust as needed
IMG_SIZE = (128, 128)

def setup_directories():
    """Create directory structure for Arabic letters"""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    
    for letter in ARABIC_LETTERS:
        letter_dir = os.path.join(BASE_DIR, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
            print(f"Created directory: {letter_dir}")
    
    # Also create directories for control characters
    for control in ['space', 'del', 'nothing']:
        control_dir = os.path.join(BASE_DIR, control)
        if not os.path.exists(control_dir):
            os.makedirs(control_dir)

def collect_arabic_sign_data(letter, num_samples=1000):
    """
    Collect sign language data for a specific Arabic letter
    
    Args:
        letter: Arabic letter to collect data for
        num_samples: Number of samples to collect
    """
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    save_dir = os.path.join(BASE_DIR, letter)
    count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')]) if os.path.exists(save_dir) else 0
    
    print(f"\n{'='*50}")
    print(f"Collecting data for: {letter}")
    print(f"Current count: {count}/{num_samples}")
    print(f"Press SPACE to capture, 'q' to quit, 's' to skip")
    print(f"{'='*50}\n")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Warning: Failed to read frame from camera")
            break
        
        # Flip frame for mirrored effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        
        # Display information
        cv2.putText(frame, f"Letter: {letter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}/{num_samples}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | Q: Quit | S: Skip", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw bounding box area
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (50, 150), (w-50, h-150), (255, 0, 0), 2)
        cv2.putText(frame, "Position hand here", (50, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Arabic Sign Language Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            # Crop and resize image
            cropped = frame[150:h-150, 50:w-50]
            if cropped.size > 0:
                resized = cv2.resize(cropped, IMG_SIZE)
                filename = os.path.join(save_dir, f"{letter}_{count:04d}.jpg")
                cv2.imwrite(filename, resized)
                count += 1
                print(f"✓ Captured: {count}/{num_samples} - {filename}")
        elif key == ord('q'):
            print(f"\n⚠️  Stopped collection for {letter}")
            print(f"   Collected: {count}/{num_samples} samples")
            break
        elif key == ord('s'):
            print(f"⏭️  Skipping {letter}")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Completed: {count} samples for {letter}\n")
    return count

def collect_all_letters():
    """Collect data for all Arabic letters"""
    setup_directories()
    
    print("\n" + "="*60)
    print("ARABIC SIGN LANGUAGE DATA COLLECTION")
    print("="*60)
    print(f"Total letters to collect: {len(ARABIC_LETTERS)}")
    print(f"Samples per letter: {NUM_SAMPLES_PER_LETTER}")
    print(f"Total samples needed: {len(ARABIC_LETTERS) * NUM_SAMPLES_PER_LETTER}")
    print("="*60 + "\n")
    
    input("Press ENTER to start data collection...")
    
    for i, letter in enumerate(ARABIC_LETTERS, 1):
        print(f"\n[{i}/{len(ARABIC_LETTERS)}] Processing letter: {letter}")
        collect_arabic_sign_data(letter, NUM_SAMPLES_PER_LETTER)
        
        if i < len(ARABIC_LETTERS):
            response = input(f"\nContinue to next letter? (y/n): ")
            if response.lower() != 'y':
                print("Data collection stopped by user.")
                break
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    # Option 1: Collect data for all letters
    # collect_all_letters()
    
    # Option 2: Collect data for specific letter
    setup_directories()
    letter = input("Enter Arabic letter to collect data for: ")
    if letter in ARABIC_LETTERS:
        collect_arabic_sign_data(letter, NUM_SAMPLES_PER_LETTER)
    else:
        print(f"❌ Invalid letter. Please use one of: {ARABIC_LETTERS}")








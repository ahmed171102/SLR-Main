import mediapipe as mp
import numpy as np

# Initialize Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For the CURRENT Arabic MLP, we still need exactly 63 landmarks (21 points * 3 coords).
# We'll extract only the hand landmarks for now to maintain compatibility with the MLP,
# but we are using the Holistic pipeline so we can easily swap to the temporal model later.
def process_frame(image_rgb: np.ndarray) -> np.ndarray:
    """
    Processes an RGB image and returns a normalized feature vector.
    For MLP compatibility, returns 63 hand features if a hand is found, else None.
    """
    # Performance: mark the image as not writeable
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True

    # Fallback to MediaPipe Hands format for the MLP
    # Prioritize right hand, then left hand
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
    elif results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
    else:
        return None

    # Extract 21 points * 3 coordinates = 63 features
    features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return features.reshape(1, -1)

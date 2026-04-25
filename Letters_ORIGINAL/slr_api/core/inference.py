import os
import tensorflow as tf
import numpy as np
from .config import MODEL_PATH, CLASS_LABELS, MIN_CONFIDENCE

# Configure GPU memory growth to prevent crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# Load the model
try:
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        model = None
    else:
        # Load MLP
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(features: np.ndarray):
    """
    Runs inference on the extracted features.
    Returns (predicted_label, confidence)
    """
    if model is None:
        return None, 0.0

    # Ensure float32
    input_tensor = tf.cast(features, tf.float32)
    
    # Predict
    prediction = model.predict(input_tensor, verbose=0)[0]
    
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])
    
    if confidence < MIN_CONFIDENCE:
        return None, confidence
        
    return CLASS_LABELS[class_idx], confidence

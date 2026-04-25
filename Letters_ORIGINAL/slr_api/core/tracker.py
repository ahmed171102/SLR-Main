import time
from collections import deque
from .config import STABILIZATION_WINDOW_SIZE, STABILIZATION_THRESHOLD, HOLD_COOLDOWN_SECONDS

class StabilizationTracker:
    def __init__(self):
        self.buffer = deque(maxlen=STABILIZATION_WINDOW_SIZE)
        self.committed_label = None
        self.cooldown_until = 0

    def update(self, label: str, confidence: float) -> dict:
        """
        Updates the tracker with a new prediction and returns the current state.
        """
        now = time.time()
        
        # If no hand/sign detected, clear buffer and release lock
        if label is None or label == 'nothing':
            self.buffer.clear()
            self.committed_label = None
            return {
                "prediction": None,
                "confidence": 0.0,
                "status": "waiting",
                "progress_pct": 0
            }

        # If in cooldown for this specific label, ignore it
        if self.committed_label == label and now < self.cooldown_until:
            return {
                "prediction": label,
                "confidence": confidence,
                "status": "cooldown",
                "progress_pct": 100
            }

        # Add to buffer
        self.buffer.append(label)
        
        # Calculate majority
        count = self.buffer.count(label)
        progress = (count / STABILIZATION_THRESHOLD) * 100
        
        if count >= STABILIZATION_THRESHOLD and len(self.buffer) == STABILIZATION_WINDOW_SIZE:
            # Commit the label
            self.committed_label = label
            self.cooldown_until = now + HOLD_COOLDOWN_SECONDS
            self.buffer.clear()
            return {
                "prediction": label,
                "confidence": confidence,
                "status": "committed",
                "progress_pct": 100
            }
            
        return {
            "prediction": label,
            "confidence": confidence,
            "status": "stabilizing",
            "progress_pct": min(100, int(progress))
        }

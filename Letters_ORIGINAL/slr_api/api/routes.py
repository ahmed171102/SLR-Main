import cv2
import numpy as np
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.preprocessing import process_frame
from core.inference import predict
from core.tracker import StabilizationTracker

router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = StabilizationTracker()
    
    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            
            # Decode frame
            try:
                # Remove header if present (e.g., "data:image/jpeg;base64,")
                if "," in data:
                    data = data.split(",")[1]
                
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. Preprocess
                features = process_frame(rgb_frame)
                
                if features is not None:
                    # 2. Inference
                    label, conf = predict(features)
                    
                    # 3. Stabilize
                    result = tracker.update(label, conf)
                else:
                    # No hand found
                    result = tracker.update(None, 0.0)
                    
                # 4. Send JSON back
                await websocket.send_json(result)
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("Client disconnected")

@router.get("/health")
def health_check():
    return {"status": "ok"}

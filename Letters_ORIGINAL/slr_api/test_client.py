import asyncio
import websockets
import cv2
import base64
import json
import time

async def stream_camera():
    uri = "ws://localhost:8000/ws/stream"
    
    # Open local webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to SLR Server")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Compress frame to reduce bandwidth
                frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # Convert to base64
                b64_str = base64.b64encode(buffer).decode('utf-8')
                
                # Send to server
                start_time = time.time()
                await websocket.send(b64_str)
                
                # Receive prediction
                response = await websocket.recv()
                data = json.loads(response)
                
                latency = int((time.time() - start_time) * 1000)
                
                # Display status in console
                if data.get("prediction"):
                    print(f"[{latency}ms] {data['prediction']} ({data['confidence']:.2f}) | {data['status']} | {data['progress_pct']}%")
                else:
                    print(f"[{latency}ms] No hand")
                
                # Avoid flooding
                await asyncio.sleep(0.03) # ~30fps
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    asyncio.run(stream_camera())

import cv2
import time
import threading
from queue import Queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()

# Configuration
encode_every = 2
jpeg_quality = 60
model = YOLO("yolo11n.engine")

# Global State
frame_queue = Queue(maxsize=10)
state_lock = threading.Lock()          # Lock for thread-safe global access
current_count = 0
current_fps = 0.0

def capture_and_detect():
    global current_count, current_fps
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, imgsz=640, verbose=False, persist=True)

        curr_time = time.time()

        
        with state_lock:
            current_count = len(results[0].boxes) if results[0].boxes is not None else 0
            current_fps = 1.0 / (curr_time - prev_time + 1e-6)

        prev_time = curr_time

        annotated_frame = results[0].plot()  
        cv2.putText(
            annotated_frame, f"FPS: {current_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA  # Fix 2
        )

        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(annotated_frame)

        time.sleep(0.005)

threading.Thread(target=capture_and_detect, daemon=True).start()

def generate_frames():
    frame_idx = 0
    last_jpeg = None

    while True:
        if not frame_queue.empty():
            while frame_queue.qsize() > 1:
                frame_queue.get_nowait()

            frame = frame_queue.get()
            frame_idx += 1

            if frame_idx % encode_every == 0 or last_jpeg is None:
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    last_jpeg = buf.tobytes()

            if last_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')
        else:
            time.sleep(0.01)

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/count")
def get_count():
    with state_lock:  #  read globals under lock
        return {"count": current_count, "fps": round(current_fps, 2)}
 
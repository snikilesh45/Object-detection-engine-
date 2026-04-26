from ultralytics import YOLO
import threading
import cv2
import time

model = YOLO("yolo11n.engine")

frame_buffers = {}
results_buffers = {}
latency_buffer = {}   # cam_id -> latest inference latency (ms)
fps_buffer = {}       # cam_id -> latest display FPS

cams = [0, 1]


def capture(cam_id):
    cap = cv2.VideoCapture(cam_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_buffers[cam_id] = frame


def inference_loop():
    while True:
        if len(frame_buffers) < len(cams):
            continue

        frames = []
        cam_ids = []

        for cam_id in cams:
            frame = frame_buffers.get(cam_id)
            if frame is not None:
                frames.append(frame)
                cam_ids.append(cam_id)

        if not frames:
            continue

        # Measure batch inference latency
        t_start = time.perf_counter()
        results = model(frames, imgsz=640, verbose=False) # Batch inference improves GPU utilization by processing multiple frames simultaneously
        latency_ms = (time.perf_counter() - t_start) * 1000

        for cam_id, res in zip(cam_ids, results):
            results_buffers[cam_id] = res
            latency_buffer[cam_id] = latency_ms / len(cam_ids)  # per-camera share


def display(cam_id):
    fps = 0
    frame_count = 0
    fps_timer = time.time()

    while True:
        frame = frame_buffers.get(cam_id)
        result = results_buffers.get(cam_id)

        if frame is None or result is None:
            continue

        annotated = result.plot()

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        latency_ms = latency_buffer.get(cam_id, 0.0)

        # Overlay FPS
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Overlay latency
        cv2.putText(
            annotated,
            f"Latency: {latency_ms:.1f} ms",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(f"Cam {cam_id}", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    threads = []

    for cam in cams:
        t = threading.Thread(target=capture, args=(cam,))
        t.start()
        threads.append(t)

    t_inf = threading.Thread(target=inference_loop)
    t_inf.start()
    threads.append(t_inf)

    for cam in cams:
        t = threading.Thread(target=display, args=(cam,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

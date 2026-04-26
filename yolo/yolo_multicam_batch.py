import cv2
import threading
import time
from ultralytics import YOLO

model = YOLO("yolo11n.engine")


def run_camera(cam_id):
    cap = cv2.VideoCapture(cam_id)

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, verbose=False)
        frame = results[0].plot()

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Draw FPS on frame
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(f"Camera {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def main():
    cams = [0, 1]  # use 2 cameras or video files

    threads = []
    for cam in cams:
        t = threading.Thread(target=run_camera, args=(cam,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
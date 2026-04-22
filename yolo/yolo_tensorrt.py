import cv2
import time
import torch
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        
        self.model = YOLO(model_path)
        

    def detect(self, frame):
        return self.model(frame, verbose=False)

    def draw(self, results):
        return results[0].plot()


def yolo_webcam():
    detector = YOLODetector("yolo11n.engine")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       

        start = time.perf_counter()
        results = detector.detect(frame)
        end = time.perf_counter()

        print(f"Inference: {(end - start):.4f}s")

        frame = detector.draw(results)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLO Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_webcam()

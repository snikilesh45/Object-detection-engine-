import cv2
import time
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda") # "cuda" for utilising gpu and "cpu" for utilising cpu
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results

    def draw(self, frame, results):
        return results[0].plot()


def yolo_webcam():
    detector = YOLODetector("yolo11n.pt")
    
    cap = cv2.VideoCapture(0)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame=cv2.resize(frame,(640,480))
        # YOLO inference with timing
        start=time.time()
        results = detector.detect(frame)
        end=time.time()
        print(f"Inference time :{end-start:.4f}s")
        # Use the class draw method (Ultralytics default colors and labels)
        frame = detector.draw(frame, results)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLO Webcam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_webcam()

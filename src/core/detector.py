import cv2
import time
import torch
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        
        self.model = YOLO(model_path)
        

    def detect(self, frame):
        return self.model(frame,imgsz=640, verbose=False)

    def draw_manual(self,frame,results):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf=float(box.conf)
                if conf<0.5:
                    continue
                # Get coordinates (top-left x, y and bottom-right x, y)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to pixels

                # Draw the rectangle (Green box, thickness of 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                
                cls = int(box.cls)
                label_name=self.model.names[cls]
                label = f"{label_name}{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame


def yolo_webcam():
    detector = YOLODetector("yolo11n.engine")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = 0
    frame_count=0
    fps_list=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       

        start = time.perf_counter()
        results = detector.detect(frame)
        end = time.perf_counter()
        frame_count+=1
        if frame_count%30==0:
            print(f"Inference: {(end - start):.4f}s")
        

        frame = detector.draw_manual(frame,results)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        fps_list.append(fps)
        if len(fps_list) >10:
            fps_list.pop(0)
        avg_fps=sum(fps_list)/len(fps_list)    

        cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLO Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_webcam()

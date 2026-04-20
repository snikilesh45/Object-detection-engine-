import cv2
import time
from ultralytics import YOLO

def yolo_webcam():
    model = YOLO("yolo11n.pt")  
    colour=({'person':(0,0,0),'laptop':(255,0,0),'dog':(0,0,255)})
    default_colour=(255,255,255)
            

    cap = cv2.VideoCapture(0)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)
        object_count=0
        # Draw detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Filter low confidence 
                if conf < 0.5:
                    continue
                object_count+=1    
                color=colour.get(label,default_colour)
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        count=f"objects:{object_count}"
        cv2.putText(frame,count,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(128,128,128),2)
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

import cv2
import time
import threading
import queue
from ultralytics import YOLO

FRAME_QUEUE_SIZE = 5
LINE_Y = 300  # horizontal line position

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

    def draw(self, results):
        return results[0].plot()


frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
stop_flag = False
fps_list = []
seen_ids = set()

#  Line-crossing counter globals
track_history = {}
count = 0
counted_ids = set()


#  Thread 1: Capture
def capture_frames():
    global stop_flag
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.001)

    cap.release()


#  Thread 2: Inference + Display
def process_frames():
    global stop_flag, seen_ids, track_history, count, counted_ids
    detector = YOLODetector("yolo11n.engine")

    prev_time = time.time()

    frame_count = 0
    start_time = time.time()

    while not stop_flag:
        frame = frame_queue.get()

        inference_start = time.perf_counter()
        results = detector.detect(frame)
        inference_time = time.perf_counter() - inference_start

        frame = detector.draw(results)

        #  Draw the counting line
        cv2.line(frame, (0, LINE_Y), (640, LINE_Y), (0, 255, 255), 2)

        #  Accumulate unique tracked object IDs
        for r in results:
            if r.boxes.id is not None:
                for obj_id in r.boxes.id:
                    seen_ids.add(int(obj_id))

        #  Line-crossing detection (top → bottom)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy
            ids = results[0].boxes.id

            for box, obj_id in zip(boxes, ids):
                obj_id = int(obj_id)

                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) // 2

                # Get previous position
                prev_y = track_history.get(obj_id, None)

                # Update history
                track_history[obj_id] = center_y

                # Check crossing (top → bottom)
                if prev_y is not None:
                    if prev_y < LINE_Y and center_y >= LINE_Y:
                        if obj_id not in counted_ids:
                            count += 1
                            counted_ids.add(obj_id)

        # Rolling FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        fps_list.append(fps)
        if len(fps_list) > 10:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # Real FPS counter
        frame_count += 1
        if time.time() - start_time >= 1:
            print("Real FPS:", frame_count)
            frame_count = 0
            start_time = time.time()

        #  Overlays — stacked vertically, no overlap
        cv2.putText(frame, f"FPS: {int(fps)} | Avg: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Unique Objects: {len(seen_ids)}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Threaded YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

    cv2.destroyAllWindows()


def main():
    t1 = threading.Thread(target=capture_frames)
    t2 = threading.Thread(target=process_frames)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    main()

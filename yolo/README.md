# YOLO Real-Time Webcam Object Detection

A simple, class-based Python implementation for real-time object detection using **YOLOv11** via webcam with GPU acceleration support.

---

## Features

- Real-time detection from your webcam feed
- GPU acceleration with CUDA support (auto-falls back to CPU)
- Performance metrics with FPS counter and inference timing
- Built-in visualization using Ultralytics' native plotting
- Modular class structure for easy integration into other projects

---

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- PyTorch (with CUDA for GPU support)

---

## Usage

```bash
python yolo_webcam.py
```

---

## Code Structure

| Component | Description |
|---|---|
| `YOLODetector` | Wrapper class for YOLO model initialization, detection, and drawing |
| `detect()` | Runs inference on a frame and returns results |
| `draw()` | Renders bounding boxes using Ultralytics' native plotting |
| `yolo_webcam()` | Main loop handling video capture, timing, and display |

---

## GPU Setup

The model automatically moves to GPU if CUDA is available:

```python
self.model.to("cuda")  # Inside YOLODetector.__init__
```

---

## Performance (After Threading + TensorRT)

| Metric | Value |
|------|------|
| Real FPS (throughput) | 29–31 FPS |
| Inference Time | 5–6 ms |
| CPU Usage | 1–2% |
| Accuracy | ~0.96 |

### Notes
- Real FPS measured using frame count per second (accurate throughput)
- Instant/average FPS may show higher values due to asynchronous pipeline behavior

## Key Insights

- TensorRT reduced inference time by ~2x
- Threading improved system responsiveness and reduced CPU usage
- Bottleneck shifted from model → display and I/O (webcam + rendering)
- Real-time systems require measuring throughput, not loop speed

## Threaded Pipeline

To improve real-time performance, the system uses a producer-consumer model:

- Thread 1: Captures frames from webcam
- Thread 2: Performs inference and displays output
- Queue: Synchronizes frames between threads

### Benefits
- Reduced lag
- Better GPU utilization
- Lower CPU usage
- Improved responsiveness

 ## FPS Clarification

Two types of FPS are observed:

- **Real FPS**: Number of frames processed per second (~30 FPS)
- **Instant FPS**: Loop speed, which may appear higher due to asynchronous processing

Only real FPS represents actual system performance.

## Limitations

- Webcam limits throughput to ~30 FPS
- `cv2.imshow()` introduces rendering overhead
- System is currently display-bound, not compute-bound



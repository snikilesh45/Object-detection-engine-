# YOLO Real-Time Webcam Object Detection

A class-based Python implementation for real-time object detection using **YOLOv11** via webcam, with TensorRT acceleration, ByteTrack tracking, line-crossing counting, and multi-mode streaming.

---

## Features

- Real-time detection from webcam feed (~30 FPS local)
- TensorRT (FP16) acceleration — ~2× inference speedup over PyTorch baseline
- Threaded producer-consumer pipeline (capture + inference decoupled)
- ByteTrack object tracking with persistent IDs
- Event-based line-crossing counter (no duplicate counts)
- Dual streaming: FastAPI MJPEG (~11 FPS) and WebRTC (~21 FPS)
- GPU auto-detection with CPU fallback
- Low CPU usage (~1–2%)

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
|-----------|-------------|
| `YOLODetector` | Wrapper class for model initialization, inference, and visualization |
| `detect()` | Runs inference on a single frame and returns results |
| `draw()` | Renders bounding boxes using Ultralytics' native plotting |
| `yolo_webcam()` | Main loop — handles capture, threading, timing, and display |

---

## Threaded Pipeline

The system uses a producer-consumer model to decouple frame capture from inference:

- **Thread 1 (Producer)** — Captures frames from webcam and pushes to queue
- **Thread 2 (Consumer)** — Pulls frames, runs inference, renders output
- **Queue** — Bounded; drops stale frames to prioritize latency over completeness

**Benefits:** reduced lag, better GPU utilization, lower CPU usage, improved responsiveness.

---

## GPU Setup

The model automatically moves to GPU if CUDA is available:

```python
self.model.to("cuda")  # Inside YOLODetector.__init__
```

Falls back to CPU if CUDA is unavailable.

---

## Tracking & Counting

- Real-time object tracking via **ByteTrack** — assigns persistent IDs across frames
- Line-crossing counter fires on crossing events only (not per-frame)
- Tracking IDs prevent duplicate counts
- Supports multi-object, multi-class counting

---

## Streaming

Two streaming modes are available:

| Method | FPS | Notes |
|--------|-----|-------|
| FastAPI (MJPEG) | ~11–12 | Simple; bottlenecked by per-frame JPEG encoding |
| WebRTC | ~21 | Low latency; eliminates encoding overhead |

Switching from MJPEG → WebRTC recovered ~10 FPS by removing the per-frame encoding bottleneck.

---

## Performance

| Metric | Value |
|--------|-------|
| Real FPS (local, OpenCV) | 29–31 FPS |
| Inference Time | 5–6 ms |
| CPU Usage | 1–2% |
| Accuracy (mAP) | ~0.96 |

**FPS note:** Real FPS (frames processed per second) is the accurate throughput metric. Instant/loop FPS may read higher due to asynchronous pipeline behavior.

---
## Batch Inference

- Supports processing multiple frames in a single inference call
- Improves GPU utilization and throughput
- Reduces per-frame overhead

### Tradeoff
- May increase latency in some cases (not observed in this setup)

## Limitations

- Webcam hardware caps throughput at ~30 FPS
- `cv2.imshow()` introduces rendering overhead
- System is I/O-bound, not compute-bound — further model optimization yields diminishing returns
- Small objects have lower confidence due to resolution constraints (640px input)

# Real-Time Object Detection Engine (YOLO + TensorRT)

A real-time object detection system built with YOLOv11, optimized using TensorRT, and enhanced with a threaded pipeline, ByteTrack-based tracking, and multi-mode streaming.

## Features

- Real-time webcam object detection (~30 FPS local)
- TensorRT acceleration (FP16) — ~2× inference speedup
- Threaded pipeline (capture + inference decoupled)
- ByteTrack-based object tracking with persistent IDs
- Line-crossing counter for entry/exit analytics
- Dual streaming: FastAPI MJPEG (~11 FPS) and WebRTC (~21 FPS)
- Low CPU usage (~1–2%)

---

## Architecture

```
Camera → Frame Queue → TensorRT Inference (GPU) → ByteTrack Tracking → Line Counter
                                                                             │
                                              ┌──────────────────────────────┘
                                              │
                                    ┌─────────┴──────────┐
                             OpenCV Display         Streaming
                              (~30 FPS)        ┌─────────┴──────────┐
                                           FastAPI MJPEG        WebRTC
                                             (~11 FPS)          (~21 FPS)
```

- **Camera thread** captures frames asynchronously and pushes to a bounded queue
- **Frame queue** drops stale frames to maintain real-time behavior (latency over completeness)
- **TensorRT inference** runs on GPU (RTX 4050) at FP16 precision — ~5–6 ms per frame
- **ByteTrack** assigns persistent IDs across frames without re-identification overhead
- **Line counter** triggers counts only on crossing events, avoiding duplicates
- **Output** branches to local OpenCV display or network streaming (MJPEG / WebRTC)

---

## Performance

| Mode | FPS | Inference | CPU |
|------|-----|-----------|-----|
| OpenCV (local) | ~30 | 5–6 ms | ~1–2% |
| FastAPI (MJPEG) | ~11 | 5–6 ms | higher |
| WebRTC | ~21 | 5–6 ms | low |

> FPS is bounded by I/O and display overhead, not model inference.

---

## System Design & Tradeoffs

### Design Decisions

**TensorRT Inference**
- Reduced inference latency from ~10 ms → ~5–6 ms (~2× improvement)
- Enables real-time performance on a consumer GPU (RTX 4050)

**Threaded Pipeline**
- Decouples frame capture from inference
- Prevents blocking; reduces CPU usage to ~1–2%

**Frame Queue with Drop Strategy**
- Discards stale frames to prioritize latency over completeness
- Keeps the pipeline from falling behind under load

**ByteTrack + Event-Based Counting**
- Tracking provides persistent object IDs across frames
- Counting fires only on line-crossing events — no duplicate counts

### Streaming Tradeoffs

**MJPEG (FastAPI)**
- Simple to implement
- Per-frame JPEG encoding creates CPU bottleneck
- Caps at ~11 FPS

**WebRTC**
- Eliminates per-frame encoding overhead
- Achieves ~21 FPS with lower latency
- More complex to set up

Switching from MJPEG → WebRTC recovered ~10 FPS and significantly reduced streaming CPU usage.

### Bottleneck Evolution

| Stage | Bottleneck |
|-------|------------|
| Baseline | Model inference |
| After TensorRT | Pipeline throughput |
| After threading | I/O / display |
| After WebRTC | Balanced system |

> Optimizing the model alone is insufficient — real-time performance depends on the full pipeline.

### Performance vs Accuracy Tradeoff

- Higher resolution → better small-object detection, lower FPS
- Lower resolution → faster inference, reduced accuracy
- Final config (640 input): mAP ~0.96 at ~30 FPS local

### Limitations

- Webcam hardware caps throughput at ~30 FPS
- Streaming adds overhead even with WebRTC
- Small objects have lower confidence due to resolution constraints

---

## Optimizations Applied

- Converted YOLO model to TensorRT (FP16)
- Fixed input resolution (640) for consistent inference shape
- Implemented threaded capture + inference pipeline
- Bounded queue with frame dropping for real-time priority
- Replaced MJPEG with WebRTC to eliminate encoding bottleneck

---

## Scaling Considerations

| Cameras | Expected FPS |
|---------|-------------|
| 1 | ~30 (local) |
| 2 | ~15–20 per stream |
| 3+ | Requires batching or multi-GPU |

**Future Improvements**
- Batch inference across multiple streams
- Asynchronous GPU execution via CUDA streams
- Load balancing across devices

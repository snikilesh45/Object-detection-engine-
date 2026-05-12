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

## Dependencies

### Core
pip install -r requirements.txt

### GPU (optional)
pip install -r requirements-gpu.txt

### TensorRT
TensorRT must be installed separately due to CUDA and system dependencies.
Refer to NVIDIA documentation.

## Running with Docker

```bash
# Build
docker compose build

# Run (requires NVIDIA Container Toolkit)
docker compose up
```

> GPU passthrough requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


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




### Notes

- **Camera thread** captures frames asynchronously and pushes to a bounded queue
- **Frame queue** drops stale frames to maintain real-time behavior (latency over completeness)
- **TensorRT inference** runs on GPU (RTX 4050) at FP16 precision — ~5–6 ms per frame
- **ByteTrack** assigns persistent IDs across frames without re-identification overhead
- **Line counter** triggers counts only on crossing events, avoiding duplicates
- **Output** branches to local OpenCV display or network streaming (MJPEG / WebRTC)
- WebRTC is used for efficient streaming compared to MJPEG
  
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

# Multi-Camera Scaling Experiment

Real-world test of running the detection pipeline on 1 vs 2 simultaneous webcam streams, measuring GPU load, per-stream FPS, and confidence impact.

---

## Setup

| | Camera 1 | Camera 2 |
|---|---|---|
| Type | Built-in laptop webcam | External USB (1080p, 60 FPS native) |
| Resolution | 640×480 | 1080p (downscaled to 640 for inference) |

Both streams share a single RTX 4050 GPU. Each stream runs its own threaded pipeline with independent inference.

---

## Results

### Single Camera

| Metric | Value |
|--------|-------|
| GPU Usage | ~50% |
| FPS (OpenCV) | ~30 |
| Confidence (mAP) | ~0.96 |

### Dual Camera

| Metric | Camera 1 (Built-in) | Camera 2 (1080p USB) |
|--------|--------------------|-----------------------|
| GPU Usage (combined) | ~71% | ~71% |
| FPS | ~20 | ~31 |
| Confidence (mAP) | ~0.94 | ~0.94 |

---

## Observations

- GPU usage scaled from **~50% → ~71%** when adding a second stream — the GPU is not saturated and can support additional cameras
- Camera 1 (built-in) dropped from ~30 → ~20 FPS due to shared GPU contention and its lower native frame rate
- Camera 2 (1080p USB) maintained ~31 FPS, likely benefiting from a higher-quality input signal and stable USB bandwidth
- Confidence dropped slightly (~0.96 → ~0.94) across both streams — consistent with increased pipeline load and potential frame drops under contention
- The bottleneck at 2 cameras shifts toward **GPU memory bandwidth and I/O scheduling**, not raw compute

---

## Takeaway

A single RTX 4050 can handle 2 simultaneous streams with acceptable performance. Scaling beyond 2 cameras would require batched inference or a dedicated GPU per stream to maintain ~30 FPS and full accuracy.


## Batch Inference Results

| Metric | Before | After |
|------|--------|-------|
| FPS (per stream) | 21–30 | 71–80 |
| Latency | 6–10 ms | 3–4.8 ms |
| GPU Usage | ~72% | ~96% |

### Key Observations

- GPU utilization increased significantly with batching
- Throughput improved across all streams
- Latency decreased due to reduced per-inference overhead
- Performance became more balanced across cameras

### Insight

Batch inference improves system efficiency by:
- minimizing kernel launch overhead
- maximizing parallel GPU execution
- reducing scheduling imbalance

This demonstrates that system-level optimization can outperform per-stream optimization.

**Future Improvements**

- Asynchronous GPU execution via CUDA streams
- Load balancing across devices

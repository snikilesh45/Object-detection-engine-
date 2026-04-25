# Performance Benchmarks

## Hardware
- GPU: NVIDIA RTX 4050
- CPU: Intel i5 (12th Gen)
- Input: Webcam (640×480)

---

## Results

###  PyTorch (Baseline)

| Metric | Value |
|------|------|
| FPS | 27–40 |
| Inference Time | ~9.9 ms |
| Accuracy | ~0.94 |

---

###  TensorRT (FP16)

| Metric | Value |
|------|------|
| FPS | 32–59 |
| Inference Time | ~5.3 ms |
| Accuracy | ~0.92–0.94 |

---

###  TensorRT + Optimized Pipeline

| Metric | Value |
|------|------|
| Real FPS (throughput) | **29–31 FPS** |
| Inference Time | **5–6 ms** |
| CPU Usage | **1–2%** |
| Accuracy | **~0.96** |

---

## Observations

- TensorRT reduced inference time by approximately **2x**
- Raw FPS increased, but not proportionally due to system overhead
- After optimization, **pipeline bottleneck shifted from model → I/O and display**
- Threading improved responsiveness and reduced CPU usage significantly

---

## Important Note on FPS

Two different FPS values can be observed:

- **Instant FPS**: Calculated per loop iteration (can spike up to 100+)
- **Real FPS**: Measured using frame count per second (~30 FPS)

Only **real FPS represents actual system throughput**

---

## Bottleneck Analysis

| Stage | Bottleneck |
|------|-----------|
| PyTorch | Model inference |
| TensorRT | Partial pipeline |
| Threaded + Optimized | Display + Webcam I/O |

---

## Key Insight

Optimizing the model (TensorRT) improves inference speed,  
but overall system performance depends on:

- Frame capture (webcam)
- Data transfer
- Rendering (`cv2.imshow`)
- Pipeline design

Final system is **I/O-bound, not compute-bound**

---

## Conclusion

- TensorRT significantly improves inference performance
- Threading enables efficient resource utilization
- Real-time performance is limited by external I/O, not the model

## Day 6 Update

- Added object tracking and line-crossing counter
- No significant impact on FPS (~30 FPS stable)
- CPU usage remains low (1–2%)

### Observation
Tracking and counting add minimal overhead due to efficient pipeline design

## Streaming Benchmarks (Day 7)

| Method | FPS | Latency | Notes |
|------|------|---------|------|
| MJPEG (FastAPI) | ~11–12 | High | Encoding bottleneck |
| WebRTC | ~21 | Low | Efficient streaming |

### Observation
- MJPEG is CPU-bound due to per-frame encoding
- WebRTC reduces overhead and improves smoothness
- Model inference time remains unchanged (~5–6 ms)

## Key Insight (Day 7)

Optimizing inference alone does not guarantee better system performance.

Streaming architecture plays a critical role:
- MJPEG → simple but inefficient
- WebRTC → complex but performant

Final system bottleneck shifted from:
Model → Pipeline → Streaming architecture

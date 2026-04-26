# Performance Benchmarks

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4050 |
| CPU | Intel Core i5 (12th Gen) |
| Input | Webcam (640×480) |

---

## Inference Benchmarks

| Mode | FPS | Inference Time | Accuracy |
|------|-----|----------------|----------|
| PyTorch (Baseline) | 27–40 | ~9.9 ms | ~0.94 |
| TensorRT (FP16) | 32–59 | ~5.3 ms | ~0.92–0.94 |
| TensorRT + Optimized Pipeline | **29–31** | **5–6 ms** | **~0.96** |

> Raw TensorRT FPS appears higher because it excludes pipeline overhead. Real throughput is measured as frames processed per second end-to-end.

---

## Streaming Benchmarks

| Method | FPS | Latency | Bottleneck |
|--------|-----|---------|------------|
| OpenCV (local) | ~30 | Low | Webcam I/O + display |
| FastAPI (MJPEG) | ~11–12 | High | Per-frame JPEG encoding |
| WebRTC | ~21 | Low | Minimal |

> Model inference time remains constant at ~5–6 ms across all streaming modes.

---

## Optimized Pipeline Metrics

| Metric | Value |
|--------|-------|
| Real FPS (throughput) | 29–31 FPS |
| Inference Time | 5–6 ms |
| CPU Usage | 1–2% |
| Tracking + Counting overhead | Negligible |

---

## FPS Measurement Note

Two FPS values can be observed:

- **Instant FPS** — calculated per loop iteration; can spike to 100+ due to async pipeline behavior
- **Real FPS** — measured as total frames processed per second (~30 FPS)

Only **real FPS represents actual system throughput** and is used throughout this document.

---

## Bottleneck Analysis

| Optimization Stage | Bottleneck |
|--------------------|------------|
| PyTorch baseline | Model inference |
| After TensorRT | Pipeline throughput |
| After threading | Display + Webcam I/O |
| After WebRTC | Balanced — no dominant bottleneck |

The system progressed from **compute-bound → I/O-bound** as each layer was optimized.

---

## Key Insights

- TensorRT reduced inference time by ~2× (~9.9 ms → ~5–6 ms)
- Threading decoupled capture and inference, dropping CPU usage to 1–2%
- ByteTrack tracking and line-crossing counter added negligible overhead
- MJPEG streaming is CPU-bound due to per-frame encoding; WebRTC eliminates this bottleneck
- Switching MJPEG → WebRTC recovered ~10 FPS in streaming throughput
- Final system is **I/O-bound, not compute-bound** — optimizing the model further yields diminishing returns

## Batch Inference Benchmark 

| Setup | FPS | Latency | GPU |
|------|------|---------|------|
| 2 Streams (no batching) | 21–30 | 6–10 ms | ~72% |
| 2 Streams (batching) | 71–80 | 3–4.8 ms | ~96% |

### Observation

Batching significantly improves:
- throughput
- GPU utilization
- system balance across streams

### Insight

Batch inference is more efficient than independent inference for multi-stream workloads.

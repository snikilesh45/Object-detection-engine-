# Real-Time Object Detection Engine (YOLO + TensorRT)

A real-time object detection system built using YOLOv11, optimized with TensorRT, and enhanced with a threaded pipeline for efficient performance.

## Features
- Real-time webcam object detection
- TensorRT acceleration (FP16)
- Threaded pipeline (capture + inference)
- Low CPU usage (1–2%)
- Stable real-time performance (~30 FPS)

## System Design & Tradeoffs

### Architecture Overview

Camera → Frame Queue → Inference (TensorRT) → Tracking → Counting → Streaming (WebRTC)

- Frames are captured asynchronously and pushed into a queue
- Inference runs on GPU using TensorRT for low latency
- Tracking assigns persistent IDs across frames
- Counting is event-based (line-crossing logic)
- Output is streamed using WebRTC for efficient real-time delivery

---

### Design Decisions

#### 1. TensorRT for Inference
- Reduced inference time from ~10 ms → ~5–6 ms (~2× improvement)
- Enabled real-time performance on consumer GPU (RTX 4050)

#### 2. Threaded Pipeline
- Decoupled frame capture from inference
- Prevented blocking and improved responsiveness
- Reduced CPU usage to ~1–2%

#### 3. Frame Queue with Dropping Strategy
- Maintains real-time behavior by discarding stale frames
- Prioritizes latency over completeness

#### 4. Tracking + Event-Based Counting
- Tracking provides persistent IDs
- Counting is triggered only when objects cross a defined line
- Avoids duplicate counting and improves reliability

---

### Streaming Tradeoffs

#### MJPEG (FastAPI)
- Simple implementation
- High CPU usage due to per-frame JPEG encoding
- Limited to ~10–12 FPS

#### WebRTC
- Efficient video streaming pipeline
- Lower latency and higher FPS (~21 FPS)
- Removes encoding bottleneck

**Conclusion:**  
Switching from MJPEG → WebRTC significantly improved system performance by eliminating per-frame encoding overhead.

---

### Bottleneck Evolution

1. Initial system → Model inference bottleneck  
2. After TensorRT → Pipeline bottleneck  
3. After threading → I/O bottleneck  
4. After WebRTC → Balanced system  

**Key Insight:**  
Optimizing the model alone is not sufficient; overall system performance depends on pipeline and streaming architecture.

---

### Performance vs Accuracy Tradeoff

- Higher resolution → better small-object detection but lower FPS  
- Lower resolution → faster inference but reduced accuracy  
- Final configuration (640 input) balances accuracy (~0.96) and performance (~30 FPS local)

---

### Limitations

- Webcam limits throughput to ~30 FPS
- Streaming introduces additional overhead (even with WebRTC)
- Small objects have lower confidence due to resolution constraints

## Performance

| Mode | FPS | Inference | CPU |
|------|------|-----------|------|
| OpenCV (local) | ~30 | 5–6 ms | ~1–2% |
| FastAPI (MJPEG) | ~11–12 | 5–6 ms | higher |
| WebRTC | ~21 | 5–6 ms | low |

### Notes
- FPS limited by I/O (display/streaming), not model
- WebRTC provides better real-time performance than MJPEG

## Optimization

- Converted YOLO model to TensorRT (FP16)
- Fixed input resolution for consistent inference
- Reduced rendering overhead
- Implemented threaded pipeline
- Controlled queue size and frame dropping

### Result
- Faster inference (~2x improvement)
- Stable FPS
- Efficient resource usage

## Key Insights

- Optimizing the model alone is not enough for real-time systems
- Bottleneck shifts from model → I/O and display after optimization
- Real-time performance must be measured using throughput, not loop speed
- Threading significantly improves responsiveness and CPU efficiency




## Progress

- Day 1: OpenCV basics
- Day 2: YOLO integration
- Day 3: Performance tuning
- Day 4: TensorRT optimization
- Day 5: Threaded real-time pipeline
- Day 6: Object Tracking and line crossing counter
- Day 7: Streaming

## Object Counting (Day 6)

Implemented a line-crossing counter using object tracking.

### Features
- Persistent object IDs using tracking
- Counts objects only when crossing a virtual line
- Avoids duplicate counting

### Use Cases
- People counting
- Traffic monitoring
- Entry/exit analytics

## Streaming (Day 7)

Two streaming methods were implemented:

### MJPEG (FastAPI)
- Simple implementation
- FPS: ~11–12
- High CPU usage due to per-frame encoding

### WebRTC
- Low-latency streaming
- FPS: ~21 (stable)
- Efficient video transmission

### Key Insight
Switching from MJPEG to WebRTC significantly improves performance by eliminating per-frame encoding overhead.

## System Evolution

1. OpenCV Pipeline → ~30 FPS
2. TensorRT Optimization → Faster inference
3. Threaded Pipeline → Stable performance
4. FastAPI (MJPEG) → Deployment but slower FPS
5. WebRTC → Improved streaming performance

### Insight
Performance optimization required moving from model-level improvements to system-level architectural changes.

## Scaling Considerations

- With multiple camera streams, GPU becomes the primary bottleneck
- Each additional stream reduces available inference time per frame
- Expected behavior:
  - 1 camera → ~30 FPS (local)
  - 2 cameras → reduced FPS per stream (~15–20)
  - 3+ cameras → requires batching or multiple GPUs

### Future Improvements
- Batch inference across streams
- Asynchronous GPU execution (CUDA streams)
- Load balancing across multiple devices

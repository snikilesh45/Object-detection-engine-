# Real-Time Object Detection Engine (YOLO + TensorRT)

A real-time object detection system built using YOLOv11, optimized with TensorRT, and enhanced with a threaded pipeline for efficient performance.

## Features
- Real-time webcam object detection
- TensorRT acceleration (FP16)
- Threaded pipeline (capture + inference)
- Low CPU usage (1–2%)
- Stable real-time performance (~30 FPS)

## Demo

- Detects multiple objects in real-time using webcam
- Displays bounding boxes, labels, and FPS
- Optimized for low latency and smooth performance

## System Architecture

The system uses a producer-consumer design:

- Thread 1: Captures frames from webcam
- Thread 2: Performs inference and rendering
- Queue: Synchronizes frames between threads

### Pipeline

Capture → Queue → TensorRT Inference → Display

## Performance

| Metric | Value |
|------|------|
| Real FPS (throughput) | 29–31 FPS |
| Inference Time | 5–6 ms |
| CPU Usage | 1–2% |
| Accuracy | ~0.96 |

### Notes
- Real FPS measured using frame count per second
- Instant FPS may appear higher due to asynchronous processing

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

## Limitations

- Webcam limits throughput to ~30 FPS
- `cv2.imshow()` introduces rendering overhead
- System is currently I/O-bound, not compute-bound


## Progress

- Day 1: OpenCV basics
- Day 2: YOLO integration
- Day 3: Performance tuning
- Day 4: TensorRT optimization
- Day 5: Threaded real-time pipeline
- Day 6:Object Tracking and line crossing counter

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

## Future Work


- Multi-camera support
- FastAPI-based streaming
- Further GPU optimization

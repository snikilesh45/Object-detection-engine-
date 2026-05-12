# System Architecture

## Overview

This document describes the architecture of the real-time object detection and tracking pipeline. The system is designed for low-latency video processing with GPU-accelerated inference, multi-object tracking, and event-driven counting.

---

## High-Level Architecture

```
┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐    ┌─────────────┐
│ Camera  │───▶│ Frame Queue │───▶│ TensorRT Inference (GPU)│───▶│ ByteTrack   │
│ Thread  │    │ (Bounded)   │    │ Object Detection        │    │ Tracking    │
└─────────┘    └─────────────┘    └─────────────────────────┘    └──────┬──────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────┐
                                                              │ Line Counter│
                                                              │ (Events)    │
                                                              └──────┬──────┘
                                                                     │
                    ┌────────────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        Output Branches                       │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
    │  │ OpenCV       │  │ FastAPI      │  │ WebRTC             │  │
    │  │ Display      │  │ MJPEG Stream │  │ Stream             │  │
    │  │ (~30 FPS)    │  │ (~11 FPS)    │  │ (~21 FPS)          │  │
    │  └──────────────┘  └──────────────┘  └────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### 1. Camera Thread (Frame Acquisition)

| Aspect | Description |
|--------|-------------|
| **Responsibility** | Captures video frames asynchronously from the camera source |
| **Behavior** | Pushes frames into a bounded queue; drops stale frames to maintain real-time behavior |
| **Design Rationale** | Latency-over-completeness: ensures the pipeline always processes the most recent frame rather than accumulating backlog |

### 2. Frame Queue (Bounded Buffer)

| Aspect | Description |
|--------|-------------|
| **Type** | Thread-safe bounded queue |
| **Policy** | Drops stale (outdated) frames when the queue is full |
| **Purpose** | Decouples the camera capture thread from the inference thread; prevents memory bloat and latency spikes under load |

### 3. TensorRT Inference (GPU)

| Aspect | Description |
|--------|-------------|
| **Engine** | NVIDIA TensorRT |
| **Hardware** | GPU (NVIDIA RTX 4050) |
| **Precision** | FP16 (half-precision) |
| **Latency** | ~5–6 ms per frame |
| **Function** | Object detection — generates bounding boxes, class labels, and confidence scores |

### 4. ByteTrack (Multi-Object Tracking)

| Aspect | Description |
|--------|-------------|
| **Algorithm** | ByteTrack |
| **Key Feature** | Assigns persistent object IDs across frames without re-identification overhead |
| **Benefit** | Maintains identity consistency for objects moving through the scene, enabling reliable event triggering |

### 5. Line Counter (Event Detection)

| Aspect | Description |
|--------|-------------|
| **Trigger** | Crossing events (objects passing a defined virtual line) |
| **Deduplication** | Counts are triggered only once per crossing event, avoiding duplicate increments |
| **Output** | Incremented count + associated metadata (object ID, timestamp, direction) |

---

## Output Branches (Streaming & Display)

The processed video stream is fanned out to three independent output channels:

| Branch | Technology | Frame Rate | Use Case |
|--------|-----------|------------|----------|
| **Local Display** | OpenCV | ~30 FPS | Real-time local visualization |
| **HTTP Streaming** | FastAPI + MJPEG | ~11 FPS | Browser-based remote viewing |
| **Low-Latency Streaming** | WebRTC | ~21 FPS | Efficient network streaming (preferred over MJPEG for bandwidth efficiency) |

> **Note:** WebRTC is selected over MJPEG for network streaming due to significantly better bandwidth efficiency and lower latency.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Inference Latency | ~5–6 ms/frame |
| Local Display FPS | ~30 FPS |
| WebRTC Streaming FPS | ~21 FPS |
| FastAPI MJPEG FPS | ~11 FPS |
| GPU Precision | FP16 |

---

## Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Bounded queue with stale-frame dropping** | Guarantees real-time responsiveness; the system prioritizes low latency over processing every single frame |
| **TensorRT + FP16 on RTX 4050** | Maximizes inference throughput while maintaining detection accuracy; FP16 offers ~2× speedup over FP32 with minimal accuracy loss |
| **ByteTrack for tracking** | Provides robust multi-object tracking without the computational cost of deep re-identification models |
| **Event-driven counting** | Counts are triggered only on crossing events, eliminating duplicate counts and reducing unnecessary computation |
| **WebRTC for network streaming** | Delivers better compression and lower latency compared to MJPEG, especially over constrained networks |

---

## Data Flow Summary

```
Camera ──▶ Queue ──▶ TensorRT (GPU) ──▶ ByteTrack ──▶ Line Counter
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    │                                     │                                     │
                    ▼                                     ▼                                     ▼
              OpenCV Display                         FastAPI MJPEG                           WebRTC
              (~30 FPS)                              (~11 FPS)                              (~21 FPS)
```

---

## Threading Model

| Thread | Role | Details |
|--------|------|---------|
| **Camera Thread** | Frame capture | Asynchronous; pushes to bounded queue |
| **Inference Thread** | GPU inference + tracking + counting | Pulls from queue; runs TensorRT → ByteTrack → Line Counter |
| **Display Thread** | OpenCV rendering | Local visualization (~30 FPS) |
| **Streaming Threads** | FastAPI & WebRTC servers | Serve processed frames to remote clients |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Inference Engine | NVIDIA TensorRT |
| Object Detection Model | YOLO (implied by ByteTrack pipeline) |
| Multi-Object Tracking | ByteTrack |
| Local Display | OpenCV |
| HTTP Streaming | FastAPI + MJPEG |
| Real-Time Streaming | WebRTC |
| GPU | NVIDIA RTX 4050 |

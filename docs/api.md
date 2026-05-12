# API Reference

## Overview

This document describes the HTTP and streaming endpoints exposed by the system. The API serves annotated video streams, detection metadata, and counting analytics.

---

## Base URL

```
http://<host>:<port>/
```

> Default port is typically `8000` for FastAPI unless configured otherwise.

---

## Endpoints

### 1. MJPEG Video Stream

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/stream/mjpeg` |
| **Content-Type** | `multipart/x-mixed-replace; boundary=frame` |
| **Frame Rate** | ~11 FPS |
| **Description** | Provides a continuous MJPEG stream of processed frames with bounding boxes, track IDs, and line counter overlays. Compatible with standard `<img>` tags and OpenCV `VideoCapture`. |

#### Usage Example

```html
<!-- Browser -->
<img src="http://localhost:8000/stream/mjpeg" alt="Live Stream">
```

```python
# OpenCV
import cv2
cap = cv2.VideoCapture("http://localhost:8000/stream/mjpeg")
```

```bash
# cURL (raw stream inspection)
curl -N http://localhost:8000/stream/mjpeg
```

#### Response Format

The response is a chunked HTTP stream where each frame is sent as:

```
--frame

Content-Type: image/jpeg


<JPEG binary data>

```

---

### 2. WebRTC Offer (SDP Exchange)

| | |
|---|---|
| **Method** | `POST` |
| **Path** | `/stream/webrtc/offer` |
| **Content-Type** | `application/json` |
| **Frame Rate** | ~21 FPS |
| **Description** | Initiates a WebRTC peer connection. Client sends an SDP offer; server responds with an SDP answer. Used for low-latency, efficient streaming compared to MJPEG. |

#### Request Body

```json
{
  "sdp": "v=0
o=- 1234567890 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0
m=video 9 UDP/TLS/RTP/SAVPF 96
c=IN IP4 0.0.0.0
a=rtpmap:96 VP8/90000
a=setup:actpass
a=mid:0
a=sendrecv
a=ice-ufrag:abc123
a=ice-pwd:def456
a=fingerprint:sha-256 ...
a=candidate:...
",
  "type": "offer"
}
```

#### Response Body

```json
{
  "sdp": "v=0
o=- 9876543210 2 IN IP4 127.0.0.1
s=-
t=0 0
...",
  "type": "answer"
}
```

#### Error Responses

| Status | Description |
|--------|-------------|
| `400` | Invalid SDP format or missing required fields |
| `500` | WebRTC peer connection failure |

---

### 3. Current Count

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/count` |
| **Content-Type** | `application/json` |
| **Description** | Returns the current cumulative count of objects that have crossed the virtual line. |

#### Response Body

```json
{
  "count": 42,
  "timestamp": "2026-05-13T01:15:30.123456",
  "direction": "bidirectional"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `count` | `integer` | Total crossing events detected since system start |
| `timestamp` | `string` (ISO 8601) | Server time at which the count was recorded |
| `direction` | `string` | Counting mode: `bidirectional`, `inbound`, or `outbound` |

---

### 4. Count Reset

| | |
|---|---|
| **Method** | `POST` |
| **Path** | `/api/count/reset` |
| **Content-Type** | `application/json` |
| **Description** | Resets the line counter to zero. Useful for starting a new counting session without restarting the pipeline. |

#### Request Body

```json
{}
```

#### Response Body

```json
{
  "success": true,
  "previous_count": 42,
  "reset_at": "2026-05-13T01:16:00.000000"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `boolean` | Whether the reset operation succeeded |
| `previous_count` | `integer` | Count value before reset |
| `reset_at` | `string` (ISO 8601) | Timestamp of the reset operation |

---

### 5. System Health / Status

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/health` |
| **Content-Type** | `application/json` |
| **Description** | Returns system health status, pipeline performance metrics, and resource utilization. |

#### Response Body

```json
{
  "status": "healthy",
  "pipeline": {
    "state": "running",
    "inference_fps": 30.5,
    "inference_latency_ms": 5.2,
    "gpu_utilization_percent": 78.0,
    "queue_size": 2,
    "queue_capacity": 10,
    "dropped_frames": 15
  },
  "streaming": {
    "mjpeg_clients": 1,
    "webrtc_clients": 0,
    "display_active": true
  },
  "timestamp": "2026-05-13T01:15:30.123456"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Overall health: `healthy`, `degraded`, or `unhealthy` |
| `pipeline.state` | `string` | Pipeline state: `running`, `paused`, or `stopped` |
| `pipeline.inference_fps` | `float` | Current inference throughput |
| `pipeline.inference_latency_ms` | `float` | Average per-frame inference latency |
| `pipeline.gpu_utilization_percent` | `float` | GPU utilization percentage |
| `pipeline.queue_size` | `integer` | Current frames in the bounded queue |
| `pipeline.queue_capacity` | `integer` | Maximum queue capacity |
| `pipeline.dropped_frames` | `integer` | Cumulative stale frames dropped |
| `streaming.mjpeg_clients` | `integer` | Active MJPEG stream consumers |
| `streaming.webrtc_clients` | `integer` | Active WebRTC peer connections |
| `streaming.display_active` | `boolean` | Whether local OpenCV display is active |

---

### 6. Configuration (Get)

| | |
|---|---|
| **Method** | `GET` |
| **Path** | `/api/config` |
| **Content-Type** | `application/json` |
| **Description** | Retrieves the current system configuration including line counter geometry, confidence thresholds, and model parameters. |

#### Response Body

```json
{
  "model": {
    "engine": "tensorrt",
    "precision": "fp16",
    "input_size": [640, 480],
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45
  },
  "tracker": {
    "algorithm": "bytetrack",
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 30
  },
  "line_counter": {
    "line": [[100, 300], [500, 300]],
    "direction": "bidirectional",
    "trigger_distance": 10
  },
  "camera": {
    "source": 0,
    "resolution": [1920, 1080],
    "fps": 30
  },
  "queue": {
    "capacity": 10,
    "drop_policy": "stale"
  }
}
```

---

### 7. Configuration (Update)

| | |
|---|---|
| **Method** | `PUT` |
| **Path** | `/api/config` |
| **Content-Type** | `application/json` |
| **Description** | Updates runtime configuration. Only provided fields are modified; omitted fields retain current values. Changes take effect immediately without pipeline restart where possible. |

#### Request Body (Partial Update)

```json
{
  "line_counter": {
    "line": [[150, 400], [600, 400]],
    "direction": "inbound"
  },
  "model": {
    "confidence_threshold": 0.6
  }
}
```

#### Response Body

```json
{
  "success": true,
  "updated_fields": ["line_counter.line", "line_counter.direction", "model.confidence_threshold"],
  "applied_at": "2026-05-13T01:17:00.000000"
}
```

#### Error Responses

| Status | Description |
|--------|-------------|
| `400` | Invalid configuration value or malformed request |
| `422` | Configuration change requires pipeline restart (not supported at runtime) |

---

## Error Response Format

All errors follow a consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_SDP",
    "message": "The provided SDP offer is malformed or missing required media sections.",
    "timestamp": "2026-05-13T01:15:30.123456",
    "request_id": "req_abc123xyz"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `error.code` | `string` | Machine-readable error identifier |
| `error.message` | `string` | Human-readable description |
| `error.timestamp` | `string` (ISO 8601) | Error occurrence time |
| `error.request_id` | `string` | Unique identifier for request tracing |

---

## WebSocket Events (Future Extension)

| Event | Direction | Payload | Description |
|-------|-----------|---------|-------------|
| `count.update` | Server → Client | `{ "count": 43, "delta": 1, "track_id": 7 }` | Pushed whenever a crossing event occurs |
| `detection.frame` | Server → Client | `{ "tracks": [...], "timestamp": "..." }` | Real-time detection metadata (lightweight, no image) |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/stream/mjpeg` | Max 5 concurrent clients |
| `/stream/webrtc/offer` | Max 3 concurrent peer connections |
| `/api/*` | 100 requests per minute |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| HTTP Framework | FastAPI |
| Streaming | MJPEG (multipart/x-mixed-replace), WebRTC |
| Data Format | JSON |
| Time Format | ISO 8601 (UTC) |

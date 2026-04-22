# Object-detection-engine-
The system converts a PyTorch-based YOLO model into an optimized TensorRT engine through an ONNX intermediate, enabling low-latency, high-throughput inference suitable for real-time applications
## Optimization (TensorRT)

The model was converted to TensorRT for faster inference.

### Results:
- ~2x faster inference
- improved FPS range
- reduced latency

### Note:
Performance is still affected by webcam input and display pipeline.

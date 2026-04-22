# Performance Comparison

## Hardware
- GPU: RTX 4050
- CPU: i5 12th Gen

## Results

| Model | FPS | Inference Time |
|------|------|----------------|
| PyTorch (.pt) | 27–40 | 9.9 ms |
| TensorRT (.engine) | 32–59 | 5.3 ms |

## Observations

- TensorRT reduces inference time by ~2x
- FPS improvement is not linear due to pipeline overhead
- Bottleneck shifts from model → I/O and display

## Final Results (After Optimization)

| Model | FPS | Inference |
|------|------|-----------|
| PyTorch | 27–40 | 9.9 ms |
| TensorRT (optimized) | 30–40 | 5–6 ms |

### Key Insight
After TensorRT optimization, the bottleneck shifted from model inference to the processing pipeline (capture + display).

## Optimization Steps

- Converted model to TensorRT (FP16)
- Fixed input resolution for consistent inference
- Reduced drawing overhead
- Limited logging frequency
- Smoothed FPS calculation

### Result
- Stable FPS (30–40)
- Consistent inference time (5–6 ms)
- Balanced system performance

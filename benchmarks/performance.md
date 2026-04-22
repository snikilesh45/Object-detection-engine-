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

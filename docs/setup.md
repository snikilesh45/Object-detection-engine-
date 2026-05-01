# GPU & TensorRT Setup

## 1. Install CUDA
Install CUDA 12.1 compatible with your GPU.

## 2. Install PyTorch (GPU)
pip install -r requirements-gpu.txt

## 3. Install TensorRT

TensorRT is not installed via standard pip requirements.

### Option A (Recommended)
Download from NVIDIA:
https://developer.nvidia.com/tensorrt

Install Python bindings:
pip install nvidia-tensorrt

### Option B (System install)
Use apt (Linux) or NVIDIA installer.

## 4. Verify

python -c "import torch; print(torch.cuda.is_available())"

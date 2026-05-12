# Base: NVIDIA TensorRT runtime (CUDA 12.1 + TensorRT 8.6.1)
FROM nvcr.io/nvidia/tensorrt:23.04-py3

# Metadata
LABEL maintainer="snikilesh45@gmail.com"
LABEL description="Real-time object detection engine — YOLO11n + TensorRT + ByteTrack"

# Set non-interactive frontend to prevent apt-get from hanging on tzdata/prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Python dependencies
COPY requirements-gpu.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-gpu.txt

# Copy source
COPY src/       ./src/
COPY configs/   ./configs/
COPY models/    ./models/

# Create non-root user 
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Default command: threaded webcam pipeline
CMD ["python3", "-m", "src.pipeline.threaded"]

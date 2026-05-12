"""
Benchmark runner for the object detection pipeline.
Measures FPS, inference time, and CPU/GPU usage across modes.

Usage:
    python benchmarks/run_benchmark.py --mode tensorrt
    python benchmarks/run_benchmark.py --mode pytorch
    python benchmarks/run_benchmark.py --mode all
"""

import argparse
import time
import csv
import os
from datetime import datetime

import cv2
import psutil
import torch
import yaml
from ultralytics import YOLO

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITOR = True
except ImportError:
    GPU_MONITOR = False


# ── Config ────────────────────────────────────────────────────────────────────

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

WARMUP_FRAMES   = 30
BENCHMARK_FRAMES = 300
INPUT_SIZE      = config["model"]["input_size"]
CAM_SOURCE      = config["camera"]["source"]


# ── GPU Monitoring ────────────────────────────────────────────────────────────

def get_gpu_usage_percent():
    if not GPU_MONITOR:
        return None
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def get_gpu_memory_mb():
    if not GPU_MONITOR:
        return None
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 ** 2


# ── Core Benchmark ────────────────────────────────────────────────────────────

def run_benchmark(model_path: str, label: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Mode: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*50}")

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(CAM_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config["camera"]["buffer_size"])

    # Warm up
    print(f"Warming up ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if ret:
            model(frame, imgsz=INPUT_SIZE, verbose=False)

    # Benchmark
    print(f"Benchmarking ({BENCHMARK_FRAMES} frames)...")
    inference_times = []
    fps_readings    = []
    gpu_usages      = []
    cpu_usages      = []
    prev_time       = time.perf_counter()

    for i in range(BENCHMARK_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — stopping early.")
            break

        t0 = time.perf_counter()
        model(frame, imgsz=INPUT_SIZE, verbose=False)
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000
        fps          = 1.0 / (t1 - prev_time + 1e-9)
        prev_time    = t1

        inference_times.append(inference_ms)
        fps_readings.append(fps)
        cpu_usages.append(psutil.cpu_percent())
        if GPU_MONITOR:
            gpu_usages.append(get_gpu_usage_percent())

        if (i + 1) % 50 == 0:
            print(f"  Frame {i+1}/{BENCHMARK_FRAMES} | "
                  f"FPS: {fps:.1f} | Inference: {inference_ms:.1f} ms")

    cap.release()

    results = {
        "label":            label,
        "model":            model_path,
        "frames":           len(inference_times),
        "avg_fps":          round(sum(fps_readings) / len(fps_readings), 2),
        "min_fps":          round(min(fps_readings), 2),
        "max_fps":          round(max(fps_readings), 2),
        "avg_inference_ms": round(sum(inference_times) / len(inference_times), 2),
        "min_inference_ms": round(min(inference_times), 2),
        "max_inference_ms": round(max(inference_times), 2),
        "avg_cpu_percent":  round(sum(cpu_usages) / len(cpu_usages), 2),
        "avg_gpu_percent":  round(sum(gpu_usages) / len(gpu_usages), 2) if gpu_usages else "N/A",
        "gpu_memory_mb":    round(get_gpu_memory_mb(), 1) if GPU_MONITOR else "N/A",
    }

    # Print summary
    print(f"\nResults for [{label}]")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    return results


# ── Save Results ──────────────────────────────────────────────────────────────

def save_results(all_results: list[dict]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = f"benchmarks/results/run_{timestamp}.csv"
    os.makedirs("benchmarks/results", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved → {out_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

MODES = {
    "pytorch":  ("yolo11n.pt",     "PyTorch Baseline"),
    "tensorrt": ("models/yolo11n.engine", "TensorRT FP16"),
}

def main():
    parser = argparse.ArgumentParser(description="Object Detection Benchmark Runner")
    parser.add_argument(
        "--mode", choices=["pytorch", "tensorrt", "all"],
        default="tensorrt",
        help="Which mode to benchmark"
    )
    args = parser.parse_args()

    modes_to_run = MODES.items() if args.mode == "all" else [(args.mode, MODES[args.mode])]

    all_results = []
    for mode_key, (model_path, label) in modes_to_run:
        results = run_benchmark(model_path, label)
        all_results.append(results)

    save_results(all_results)


if __name__ == "__main__":
    main()

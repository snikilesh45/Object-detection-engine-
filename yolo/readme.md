YOLO Real-Time Webcam Object Detection:
A simple, class-based Python implementation for real-time object detection using YOLOv11 via webcam with GPU acceleration support.


Features
 
- Real-time detection from your webcam feed
 
- GPU acceleration with CUDA support (auto-falls back to CPU)
 
- Performance metrics with FPS counter and inference timing
 
- Built-in visualization using Ultralytics' native plotting
 
- Modular class structure for easy integration into other projects

Requirements
 
-Python 3.8+
 
-OpenCV
 
-Ultralytics YOLO
 
-PyTorch (with CUDA for GPU support)

Code Structure:

Components:
-YOLODetector: 
Wrapper class for YOLO model initialization, detection, and drawing
-detect():
Runs inference on a frame and returns results
-draw(): 
Renders bounding boxes using Ultralytics' native plotting
-yolo_webcam(): 
Main loop handling video capture, timing, and display

GPU Setup:
The model automatically moves to GPU if CUDA is available:
self.model.to("cuda")  # Inside YOLODetector.__init__

Performance Benchmarks:
Tested on Intel i5-12th Gen (CPU) vs NVIDIA RTX 4050 (GPU) using YOLOv11n:
 ---------------------------------------------
 |Resolution  |   CPU         |  GPU         |
 |            | FPS | ACCURACY|  FPS|ACCURACY|
 | 320x240    | 7   |  0.92   |12-40|0.93    |
 | 640x480    | 8   |  0.94   |10-51|0.92    |
 | 1280x720   | 9   |  0.95   |7-40 |0.90    |
----------------------------------------------

Observations:
-CPU performance is relatively stable across resolutions (~7–9 FPS) with slight accuracy edge at higher resolutions.
-GPU performance shows wide variance (7–51 FPS) depending on scene complexity and thermal throttling.

